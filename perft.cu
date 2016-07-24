#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "perft_bb.h"
#include <math.h>
#include <stdlib.h>

//--------------------------------------------------------------------------------------------------
//  Util functions (TODO: move these to utils.cpp/.h ?)
//--------------------------------------------------------------------------------------------------

class EventTimer {
public:
  EventTimer() : mStarted(false), mStopped(false) {
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
  }
  ~EventTimer() {
    cudaEventDestroy(mStart);
    cudaEventDestroy(mStop);
  }
  void start(cudaStream_t s = 0) { cudaEventRecord(mStart, s); 
                                   mStarted = true; mStopped = false; }
  void stop(cudaStream_t s = 0)  { assert(mStarted);
                                   cudaEventRecord(mStop, s); 
                                   mStarted = false; mStopped = true; }
  float elapsed() {
    assert(mStopped);
    if (!mStopped) return 0; 
    cudaEventSynchronize(mStop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, mStart, mStop);
    return elapsed;
  }

private:
  bool mStarted, mStopped;
  cudaEvent_t mStart, mStop;
};

// for timing CPU code : start
double gTime;
#define START_TIMER { \
    clock_t start, end; \
    start = clock();

#define STOP_TIMER \
    end = clock(); \
    gTime = (double)(end - start)/1000.0;}
// for timing CPU code : end


static void hugeMemset(void *data, uint64 size)
{
    uint8 *mem = (uint8*)data;
    const uint64 c4G = 4ull * 1024 * 1024 * 1024;

    while (size > c4G)
    {
        cudaMemset(mem, 0, c4G);

        mem += c4G;
        size -= c4G;
    }

    cudaMemset(mem, 0, size);
}

#if USE_TRANSPOSITION_TABLE == 1
    // can't make this bigger than 6, as the _simple kernel (breadth first search) gets called directly
    // breadth first search uses lot of memory and can can't hold bigger tree 
    #define GPU_LAUNCH_DEPTH 6
#else
    // do perft 10 using a single GPU call
    // bigger perfts are divided on the CPU
    #define GPU_LAUNCH_DEPTH 10
#endif

void allocAndClearMem(void **devPointer, void **hostPointer, size_t size, bool sysmem, int depth)
{
    cudaError_t res;
    void *temp = NULL;
    *devPointer = NULL;

    if (sysmem)
    {
        if (depth >= GPU_LAUNCH_DEPTH)
        {
            // plain system memory
            temp = malloc(size);
        }
        else
        {
            // try allocating in system memory
            res = cudaHostAlloc(&temp, size, cudaHostAllocMapped | cudaHostAllocWriteCombined);
            if (res != cudaSuccess)
            {
                printf("\nFailed to allocate sysmem transposition table of %d bytes, with error: %s\n", size, cudaGetErrorString(res));
            }
            res = cudaHostGetDevicePointer(devPointer, temp, 0);
            if (res != S_OK)
            {
                printf("\nFailed to get GPU mapping for sysmem hash table, with error: %s\n", cudaGetErrorString(res));
            }
        }
    }
    else
    {
        res = cudaMalloc(devPointer, size);
        if (res != cudaSuccess)
        {
            printf("\nFailed to allocate GPU transposition table of %d bytes, with error: %s\n", size, cudaGetErrorString(res));
        }
    }
    *hostPointer = temp;
    if (devPointer)
    {
        hugeMemset(devPointer, size);
    }
    else
    {
        assert(*hostPointer);
        memset(*hostPointer, 0, size);
    }
}

#if USE_TRANSPOSITION_TABLE == 1
void setupHashTables128b(TTInfo128b &tt)
{
    // size of transposition tables for each depth
    // 25 bits -> 32 million entries
    // 26 bits -> 64 million ...
    //           depth->     0        1      2      3       4       5       6       7       8       9      10      11      12      13      14      15         

    const bool  shallow[] = {true, true,  true,   true,   true,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false};

    // settings for 12 GB card, + 16 GB sysmem
#if 0
    const uint32 ttBits[] = {0,       0,    24,     28,     27,     26,     25,     25,     25,      0,      0,      0,      0,      0,      0,      0};
    const bool   sysmem[] = {true, true, false,  false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
#else
    // settings for laptop (2 GB card + 16 GB sysmem)
    const uint32 ttBits[] = {0,       0,     25,     26,     26,     25,     25,     25,     25,      0,      0,      0,      0,      0,      0,      0};
    const bool   sysmem[] = {true, true,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
#endif

    const int  sharedHashBits = 25;
    const bool  sharedsysmem = true;

    // allocate the shared hash table
    void *sharedTable, *sharedTableCPU;
    allocAndClearMem(&sharedTable, &sharedTableCPU, GET_TT_SIZE_FROM_BITS(sharedHashBits) * sizeof(HashEntryPerft128b), sharedsysmem, 9);

    memset(&tt, 0, sizeof(tt));
    for (int i = 2; i < MAX_PERFT_DEPTH; i++)
    {
        tt.shallowHash[i] = shallow[i];
        uint32 bits = ttBits[i];
        if (bits)
        {
            allocAndClearMem(&tt.hashTable[i], &tt.cpuTable[i],
                GET_TT_SIZE_FROM_BITS(bits) * (shallow[i] ? sizeof(HashKey128b) : sizeof(HashEntryPerft128b)), sysmem[i], i);
        }
        else
        {
            tt.hashTable[i] = sharedTable;
            tt.cpuTable[i] = sharedTableCPU;
            bits  = sharedHashBits;
        }
        tt.indexBits[i] = GET_TT_INDEX_BITS(bits);
        tt.hashBits[i] = GET_TT_HASH_BITS(bits);
    }
}
#endif

// TODO: avoid this global var?
TTInfo128b TransTables128b;

void initGPU()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }

    // allocate the buffer to be used by device code memory allocations
    cudaStatus = cudaMalloc(&preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s", cudaGetErrorString(cudaStatus));
        exit(0);
    }
    else
    {
        printf("\nAllocated preAllocatedBuffer of %d bytes, address: %X\n", PREALLOCATED_MEMORY_SIZE, preAllocatedBufferHost);
    }

    cudaMemset(&preAllocatedMemoryUsed, 0, sizeof(uint32));

#if USE_TRANSPOSITION_TABLE == 1
    setupHashTables128b(TransTables128b);
#endif
}

uint32 estimateLaunchDepth(HexaBitBoardPosition *pos)
{
    // estimate branching factor near the root
    double perft1 = perft_bb(pos, 1);
    double perft2 = perft_bb(pos, 2);
    double perft3 = perft_bb(pos, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
    float geoMean = sqrt((perft3/perft2) * (perft2/perft1));
    float arithMean = ((perft3/perft2) + (perft2/perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }
        
    //printf("\nEstimated branching factor: %g\n", branchingFactor);

    float memLimit = PREALLOCATED_MEMORY_SIZE / 2;  // be conservative as the branching factor can increase later

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = log(memLimit) / log (branchingFactor);

    //printf("\nEstimated launch depth: %d\n", depth);

    return depth;
}

uint64 perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, HexaBitBoardPosition *gpuBoard, uint64 *gpu_perft, void *serial_perft_stack, int launchDepth, char *dispPrefix)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];
    CMove genMoves[MAX_MOVES];
    char  dispString[128];

#if USE_TRANSPOSITION_TABLE == 1
    HashKey128b posHash128b;
    posHash128b = MoveGeneratorBitboard::computeZobristKey128b(pos);

    // check hash table
    HashEntryPerft128b *hashTable = (HashEntryPerft128b *) TransTables128b.cpuTable[depth];
    uint64 indexBits = TransTables128b.indexBits[depth];
    uint64 hashBits = TransTables128b.hashBits[depth];
    HashEntryPerft128b entry;

    if (hashTable)
    {
        entry = hashTable[posHash128b.lowPart & indexBits];
        // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
        entry.hashKey.highPart ^= entry.perftVal;
        entry.hashKey.lowPart ^= entry.perftVal;

        if ((entry.hashKey.highPart == posHash128b.highPart) && ((entry.hashKey.lowPart & hashBits) == (posHash128b.lowPart & hashBits))
            && (entry.depth == depth))
        {
            // hash hit
            return entry.perftVal;
        }
    }
#endif

    uint32 nMoves = 0;
    uint64 count = 0;

    if (depth <= GPU_LAUNCH_DEPTH)
    {
        // launch GPU perft routine
        uint64 res;
        {
            EventTimer gputime;
            gputime.start();

            cudaMemcpy(gpuBoard, pos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
            cudaMemset(gpu_perft, 0, sizeof(uint64));
            // gpu_perft is a single 64 bit integer which is updated using atomic adds by leave nodes
            #if USE_TRANSPOSITION_TABLE == 1
                    perft_bb_gpu_simple_hash << <1, 1 >> > (gpuBoard, posHash128b, gpu_perft, depth, preAllocatedBufferHost,
                                                            TransTables128b);
            #else
                perft_bb_driver_gpu <<<1, 1>>> (gpuBoard, gpu_perft, depth, serial_perft_stack, preAllocatedBufferHost, launchDepth);
            #endif

            cudaError_t err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
            gputime.stop();
#if USE_TRANSPOSITION_TABLE == 0
            if (err != S_OK) printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));
            printf("\nGPU Perft %d: %llu,   ", depth, res);
            fflush(stdout);
            printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) (((double) res/gputime.elapsed())*1000.0));
#endif
        }

        count = res;
    }
    else
    {
        //nMoves = generateBoards(pos, newPositions);
        // generate moves also so that we can print them

#if 1
         nMoves = generateMoves(pos, pos->chance, genMoves);
#else
        // Ankan TODO: there is some bug here!

        // generate all non-captures first to reduce hash table trashing
        if (pos->chance == WHITE)
        {
            ExpandedBitBoard ebb;
            ebb = MoveGeneratorBitboard::ExpandBitBoard<WHITE>(pos);
            nMoves  = MoveGeneratorBitboard::generateNonCaptures<WHITE>(&ebb, genMoves);
            nMoves += MoveGeneratorBitboard::generateCaptures<WHITE>(&ebb, &genMoves[nMoves]);
        }
        else
        {
            ExpandedBitBoard ebb;
            ebb = MoveGeneratorBitboard::ExpandBitBoard<BLACK>(pos);
            nMoves = MoveGeneratorBitboard::generateNonCaptures<BLACK>(&ebb, genMoves);
            nMoves += MoveGeneratorBitboard::generateCaptures<BLACK>(&ebb, &genMoves[nMoves]);
        }
#endif
        for (uint32 i = 0; i < nMoves; i++)
        {
            newPositions[i] = *pos;
            uint64 fakeHash = 0;

            if (pos->chance == WHITE)
                MoveGeneratorBitboard::makeMove<WHITE, false>(&newPositions[i], fakeHash, genMoves[i]);
            else
                MoveGeneratorBitboard::makeMove<BLACK, false>(&newPositions[i], fakeHash, genMoves[i]);

            char moveString[10];
            Utils::getCompactMoveString(genMoves[i], moveString);
            strcpy(dispString, dispPrefix);
            strcat(dispString, moveString);
            uint64 childPerft = perft_bb_cpu_launcher(&newPositions[i], depth - 1, gpuBoard, gpu_perft, serial_perft_stack, launchDepth, dispString);
            //printf("%s   %20llu\n", dispString, childPerft);
            //fflush(stdout);
            count += childPerft;
        }
    }

#if USE_TRANSPOSITION_TABLE == 1
    // store in hash table
    // replace only if old entry was shallower (or of same depth)
    if (hashTable && entry.depth <= depth)
    {
        HashEntryPerft128b newEntry;
        newEntry.perftVal = count;
        newEntry.hashKey.highPart = posHash128b.highPart;
        newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
        newEntry.depth = depth;

        // XOR hash part with data part for lockless hashing
        newEntry.hashKey.lowPart ^= newEntry.perftVal;
        newEntry.hashKey.highPart ^= newEntry.perftVal;

        hashTable[posHash128b.lowPart & indexBits] = newEntry;
    }
#endif
    return count;
}


// called only for bigger perfts - shows move count distribution for each move
void dividedPerft(HexaBitBoardPosition *pos, uint32 depth, int launchDepth)
{
    HexaBitBoardPosition *gpuBoard;
    uint64 *gpu_perft;
    void *serial_perft_stack;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for gpuBoard, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMalloc(&serial_perft_stack, GPU_SERIAL_PERFT_STACK_SIZE);
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for serial_perft_stack, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMalloc(&gpu_perft, sizeof(uint64));
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for gpu_perft, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    printf("\n");
    uint64 perft;
    START_TIMER
    perft = perft_bb_cpu_launcher(pos, depth, gpuBoard, gpu_perft, serial_perft_stack, launchDepth, "..");
    STOP_TIMER

#if USE_TRANSPOSITION_TABLE == 1
    printf("Perft(%02d):%20llu, time: %8g s\n", depth, perft, gTime);
    fflush(stdout);
#endif
	cudaFree(gpuBoard);
    cudaFree(gpu_perft);
    cudaFree(serial_perft_stack);    
}

int main(int argc, char *argv[])
{
    BoardPosition testBoard;
    initGPU();
    MoveGeneratorBitboard::init();

    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results
    //Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)
    //Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!
    //Utils::readFENString("r1b1kbnr/pppp1ppp/2n1p3/6q1/6Q1/2N1P3/PPPP1PPP/R1B1KBNR w KQkq - 4 4", &testBoard); // temp test

    int minDepth = 3;
    int maxDepth = 3;
    char fen[1024];
    if (argc >= 3)
    {
        strcpy(fen, argv[1]);
        maxDepth = atoi(argv[2]);
    }
    else
    {
        printf("\nUsage perft_gpu <fen> <depth> [<launchdepth>]\n");
        printf("\nAs no paramaters were provided... running default test\n");
    }

    if (strlen(fen) > 5)
    {
        Utils::readFENString(fen, &testBoard);
    }
    else
    {
        Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    }
    Utils::dispBoard(&testBoard);

    

    HexaBitBoardPosition testBB;
    Utils::board088ToHexBB(&testBB, &testBoard);
    Utils::boardHexBBTo088(&testBoard, &testBB);

    // launchDepth is the depth at which the driver kernel launches the work kernels
    // we decide launch depth based by estimating memory requirment of the work kernel that would be launched.

    // TODO: need more accurate method to estimate launch depth
    // branching factor near the root is not accurate. E.g, for start pos, at root branching factor = 20
    // and we estimate launch depth = 6.. which would seem quite conservative (20^6 = 64M)
    // at depth 10, the avg branching factor is nearly 30 and 30^6 = 729M which is > 10X initial estimate :-/
    
    // At launch depth 6, some launches for perft 9 start using up > 350 MB memory
    // 384 MB is not sufficient for computing perft 10 (some of the launches consume more than that)
    // and 1 GB is not sufficient for computing perft 11!
    
    uint32 launchDepth = estimateLaunchDepth(&testBB);
    launchDepth = min(launchDepth, 11); // don't go too high

    // for best performance without GPU hash (also set PREALLOCATED_MEMORY_SIZE to 3 x 768MB)
    // launchDepth = 6;    // ankan - test!

    if (argc >= 4)
    {
        launchDepth = atoi(argv[3]);
    }

    if (maxDepth < launchDepth)
    {
        launchDepth = maxDepth;
    }

    for (int depth = minDepth; depth <= maxDepth; depth++)
    {
        dividedPerft(&testBB, depth, launchDepth);
    }
    
    cudaFree(preAllocatedBufferHost);
    cudaDeviceReset();
    return 0;
}
