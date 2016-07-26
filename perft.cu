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

// can't make this bigger than 6, as the _simple kernel (breadth first search) gets called directly
// breadth first search uses lot of memory and can can't hold bigger tree 
#define GPU_LAUNCH_DEPTH 6

// print divided perft values (subtotals) after reaching this depth
#define DIVIDED_PERFT_DEPTH 10

// try launching multiple GPU kenerls in  parallel (on multiple GPUs)
#define ENABLE_MULTIPLE_PARALLEL_LAUNCHES 1

// no. of max parallel kernels
// didn't help much (or at all!)
#define MAX_STREAMS_PER_GPU 4
cudaStream_t cudaStream[MAX_STREAMS_PER_GPU];

// whether or not use real different streams (use default stream.. aka tail parallelism when it's 0)
#define SCHEDULE_MULTIPLE_STREAMS 0

// the stream thing didn't work well (< 10% speedup for so much pain :-/)
// try modifying the kernel to achieve better occupancy
// first implement multi-gpu search by using producer consumer approach
// 3 GPUs (three threads for launching GPU work) are the consumers.
// 1 CPU thread is the producer (produces a new item for perft)
//  -- last level of boards similar to stream kernel
// 
//  .. or maybe sismply round robin among the GPUs and sync (readback all results) at the end? (try this first!)
// getting ~2X scaling wiht 3 GPUs with the above (round robin) approach.
//  - the major problem is probably lower transposition table effectiveness (specifically for video memory transposition tables)
//  - launching more work on a GPU is likely to help?

int numGPUs = 0;

#if USE_TRANSPOSITION_TABLE == 1

// TODO: avoid these global vars?
TTInfo128b TransTables128b[MAX_GPUs];
HexaBitBoardPosition *gpuBoard[MAX_GPUs];
uint64 *gpu_perft[MAX_GPUs];

// to avoid allocating sysmem tables multiple times!
bool sysmemTablesAllocated = false;

void allocAndClearMem(void **devPointer, void **hostPointer, size_t size, bool sysmem, int depth)
{
    cudaError_t res;
    void *temp = NULL;
    *devPointer = NULL;

    if (sysmem)
    {
        if (depth >= GPU_LAUNCH_DEPTH)
        {
            if (sysmemTablesAllocated)
            {
                temp = TransTables128b[0].cpuTable[depth];
            }
            else
            {
                // plain system memory
                temp = malloc(size);
                if (!temp)
                {
                    printf("\nFailed allocating pure sysmem for transposition table!\n");
                    exit(0);
                }
            }
        }
        else
        {
            // try allocating in system memory
            if (sysmemTablesAllocated)
            {
                temp = TransTables128b[0].cpuTable[depth];
                *devPointer = TransTables128b[0].hashTable[depth];
            }
            else
            {
                res = cudaHostAlloc(&temp, size, cudaHostAllocMapped | cudaHostAllocWriteCombined /*| cudaHostAllocPortable*/);
                if (res != cudaSuccess)
                {
                    printf("\nFailed to allocate sysmem transposition table of %d bytes, with error: %s\n", size, cudaGetErrorString(res));
                    exit(0);
                }
                
                res = cudaHostGetDevicePointer(devPointer, temp, 0);
                if (res != S_OK)
                {
                    printf("\nFailed to get GPU mapping for sysmem hash table, with error: %s\n", cudaGetErrorString(res));
                    exit(0);
                }
            }
        }
    }
    else
    {
        res = cudaMalloc(devPointer, size);
        if (res != cudaSuccess)
        {
            printf("\nFailed to allocate GPU transposition table of %d bytes, with error: %s\n", size, cudaGetErrorString(res));
            exit(0);
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


// size of transposition tables for each depth
// 25 bits -> 32  million entries (512 MB)
// 26 bits -> 64  million ...     (1 GB)
// 27 bits -> 128 million ...     (2 GB)
// 28 bits -> 256 million ...     (4 GB)
//           depth->     0        1      2      3       4       5       6       7       8       9      10      11      12      13      14      15         

const bool  shallow[] = {true, true,  true,   true,   true,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false};

// settings for Titan X (12 GB card) + 16 GB sysmem
#if 0
const uint32 ttBits[] = {0,       0,    25,     28,     26,     26,     27,     25,     25,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, true, false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
#else
// settings for laptop (2 GB card + 16 GB sysmem)
const uint32 ttBits[] = {0,       0,     25,     26,     26,     25,     25,     25,     25,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, true,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
#endif

const int  sharedHashBits = 27;
const bool sharedsysmem = true;

void setupHashTables128b(TTInfo128b &tt)
{
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

    sysmemTablesAllocated = true;
}

void freeHashTables()
{
    bool sharedDeleted = false;
    for (int g = 0; g < numGPUs; g++)
    {
        cudaSetDevice(g);
        for (int i = 0; i < MAX_PERFT_DEPTH; i++)
        {
            uint32 bits = ttBits[i];
            if (bits == 0)
            {
                if (!sharedDeleted)
                {
                    // delete the shared sysmem hash table
                    free(TransTables128b[g].cpuTable[i]);
                    sharedDeleted = true;
                }
            }
            else if(sysmem[i])
            {
                if (g == 0)
                {
                    if (i >= GPU_LAUNCH_DEPTH)
                        free(TransTables128b[g].cpuTable[i]);
                    else
                        cudaFree(TransTables128b[g].hashTable[i]);
                }
            }
            else
            {
                cudaFree(TransTables128b[g].hashTable[i]);
            }
        }
    }
    cudaSetDevice(0);
    memset(TransTables128b, 0, sizeof(TransTables128b));
}

// launch work on multiple streams (or GPUs), wait for enough parallel work is done, and only then wait for streams/GPUs to finish
uint64 perft_multi_stream_launcher(HexaBitBoardPosition *pos, uint32 depth)
{
    cudaError_t cudaStatus;
    uint64 count = 0;
    CMove genMoves[MAX_MOVES];
    HashKey128b hashes[MAX_MOVES];
    HexaBitBoardPosition childPos;

    int nNewBoards = 0;     // no. of boards not found in transposition table
    int nMoves = generateMoves(pos, pos->chance, genMoves);

    HashEntryPerft128b *hashTable = (HashEntryPerft128b *) TransTables128b[0].cpuTable[depth - 1];
    uint64 indexBits = TransTables128b[0].indexBits[depth - 1];
    uint64 hashBits = TransTables128b[0].hashBits[depth - 1];

    uint64 perftResults[MAX_MOVES];

    for (int i = 0; i < nMoves; i++)
    {
        childPos = *pos;
        uint64 fakeHash = 0;

        if (pos->chance == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE, false>(&childPos, fakeHash, genMoves[i]);
        else
            MoveGeneratorBitboard::makeMove<BLACK, false>(&childPos, fakeHash, genMoves[i]);

        HashKey128b posHash128b;
        posHash128b = MoveGeneratorBitboard::computeZobristKey128b(&childPos);

#if 1
        // check in hash table
        if (hashTable)
        {
            HashEntryPerft128b entry;
            entry = hashTable[posHash128b.lowPart & indexBits];
            // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
            entry.hashKey.highPart ^= entry.perftVal;
            entry.hashKey.lowPart ^= entry.perftVal;

            if ((entry.hashKey.highPart == posHash128b.highPart) && ((entry.hashKey.lowPart & hashBits) == (posHash128b.lowPart & hashBits))
                && (entry.depth == (depth - 1)))
            {
                // hash hit
                count += entry.perftVal;
                continue;
            }
        }
#endif        
        {
            hashes[nNewBoards] = posHash128b;

            int gpu = nNewBoards % numGPUs;
            cudaStatus = cudaSetDevice(gpu);
            if (cudaStatus != cudaSuccess) printf("\nCudaSetDevice failed, error: %s\n", cudaGetErrorString(cudaStatus));

            HexaBitBoardPosition *boards = gpuBoard[gpu];
            uint64 *perfts = gpu_perft[gpu];

            cudaStatus = cudaMemcpy(&boards[nNewBoards], &childPos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) printf("\cudaMemcpy board failed, error: %s\n", cudaGetErrorString(cudaStatus));
            cudaStatus = cudaMemset(&perfts[nNewBoards], 0, sizeof(uint64));
            if (cudaStatus != cudaSuccess) printf("\cudaMemset perfts failed, error: %s\n", cudaGetErrorString(cudaStatus));

            //printf("launching kernel with board: %X, perft: %X\n", &boards[nNewBoards], &perfts[nNewBoards]);

            perft_bb_gpu_simple_hash << <1, 1 >> > (&boards[nNewBoards], posHash128b, &perfts[nNewBoards], depth - 1, preAllocatedBufferHost[gpu],
                                                    TransTables128b[gpu], true);

            nNewBoards++;
        }
    }


    for (int i = 0; i < nNewBoards; i++)
    {
        int gpu = i % numGPUs;
        uint64 *perfts = gpu_perft[gpu];

        cudaError_t cudaStatus = cudaMemcpy(&perftResults[i], &perfts[i], sizeof(uint64), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("\nFAILED getting perft result from GPU: %s\n", cudaGetErrorString(cudaStatus));
        }

        //printf("\n%d\n", perftResults[i]);
        count += perftResults[i];

#if 1
        // store in hash table
        HashKey128b posHash128b = hashes[i];

        HashEntryPerft128b oldEntry;
        oldEntry = hashTable[posHash128b.lowPart & indexBits];

        // replace only if old entry was shallower (or of same depth)
        if (hashTable && oldEntry.depth <= (depth-1))
        {
            HashEntryPerft128b newEntry;
            newEntry.perftVal = perftResults[i];
            newEntry.hashKey.highPart = posHash128b.highPart;
            newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
            newEntry.depth = (depth-1);

            // XOR hash part with data part for lockless hashing
            newEntry.hashKey.lowPart ^= newEntry.perftVal;
            newEntry.hashKey.highPart ^= newEntry.perftVal;

            hashTable[posHash128b.lowPart & indexBits] = newEntry;
        }
#endif
    }

    return count;
}


uint64 perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];
    CMove genMoves[MAX_MOVES];
    char  dispString[128];

    HashKey128b posHash128b;
    posHash128b = MoveGeneratorBitboard::computeZobristKey128b(pos);

    // check hash table
    HashEntryPerft128b *hashTable = (HashEntryPerft128b *) TransTables128b[0].cpuTable[depth];
    uint64 indexBits = TransTables128b[0].indexBits[depth];
    uint64 hashBits = TransTables128b[0].hashBits[depth];
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

    uint32 nMoves = 0;
    uint64 count = 0;

    if (depth <= GPU_LAUNCH_DEPTH)
    {
        // launch GPU perft routine
        uint64 res;
        {
            cudaMemcpy(gpuBoard[0], pos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
            cudaMemset(gpu_perft[0], 0, sizeof(uint64));
            // gpu_perft is a single 64 bit integer which is updated using atomic adds by leave nodes
            perft_bb_gpu_simple_hash <<<1, 1 >>> (gpuBoard[0], posHash128b, gpu_perft[0], depth, preAllocatedBufferHost[0],
                                                    TransTables128b[0], true);

            cudaError_t err = cudaMemcpy(&res, gpu_perft[0], sizeof(uint64), cudaMemcpyDeviceToHost);
        }

        count = res;
    }
    else
#if ENABLE_MULTIPLE_PARALLEL_LAUNCHES == 1
    if (depth == GPU_LAUNCH_DEPTH + 1)
    {
        count = perft_multi_stream_launcher(pos, depth);
    }
    else
#endif
    {

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
            uint64 childPerft = perft_bb_cpu_launcher(&newPositions[i], depth - 1, dispString);

            if (depth > DIVIDED_PERFT_DEPTH)
            {
                printf("%s   %20llu\n", dispString, childPerft);
                fflush(stdout);
            }
            count += childPerft;
        }
    }

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

    return count;
}

// called only for bigger perfts - shows move count distribution for each move
void dividedPerft(HexaBitBoardPosition *pos, uint32 depth)
{
    cudaError_t cudaStatus;

    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);
        cudaStatus = cudaMalloc(&gpuBoard[i], sizeof(HexaBitBoardPosition) * MAX_MOVES);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for gpuBoard, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(0);
        }

        cudaStatus = cudaMalloc(&gpu_perft[i], sizeof(uint64) * MAX_MOVES);
        if (cudaStatus != cudaSuccess) 
        {
            printf("cudaMalloc failed for gpu_perft, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(0);
        }
    }
    cudaSetDevice(0);

#if 0
    for (int i = 0; i < MAX_STREAMS_PER_GPU; i++)
    {
        cudaStatus = cudaStreamCreate(&cudaStream[i]);
        if (cudaStatus != cudaSuccess) printf("cudaStreamCreate failed, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
    }
#endif

    printf("\n");
    uint64 perft;
    START_TIMER
    perft = perft_bb_cpu_launcher(pos, depth, "..");
    STOP_TIMER

    printf("Perft(%02d):%20llu, time: %8g s\n", depth, perft, gTime);
    fflush(stdout);

    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);
        cudaFree(gpuBoard[i]);
        cudaFree(gpu_perft[i]);
    }
    cudaSetDevice(0);

#if 0
    for (int i = 0; i < MAX_STREAMS_PER_GPU; i++)
    {
        cudaStreamDestroy(cudaStream[i]);
    }
#endif
}
#endif


// either launch GPU routine directly (for no-hash perfts) or call recursive serial CPU routine for divided perfts
void perftLauncher(HexaBitBoardPosition *pos, uint32 depth, int launchDepth)
{
#if USE_TRANSPOSITION_TABLE == 1
    dividedPerft(pos, depth);
#else
    uint64 res;
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

    cudaMemcpy(gpuBoard, pos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
    cudaMemset(gpu_perft, 0, sizeof(uint64));

    EventTimer gputime;
    gputime.start();

    // gpu_perft is a single 64 bit integer which is updated using atomic adds by leaf nodes
    perft_bb_driver_gpu <<<1, 1 >>> (gpuBoard, gpu_perft, depth, serial_perft_stack, preAllocatedBufferHost[0], launchDepth);

    cudaError_t err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
    gputime.stop();

    if (err != S_OK) printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));
    printf("\nGPU Perft %d: %llu,   ", depth, res);
    fflush(stdout);
    printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed() / 1000.0, (uint64)(((double)res / gputime.elapsed())*1000.0));

    cudaFree(gpuBoard);
    cudaFree(gpu_perft);
    cudaFree(serial_perft_stack);
#endif
}

void initGPU(int gpu)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! error: %s\n", cudaGetErrorString(cudaStatus));
        exit(0);
    }

    // allocate the buffer to be used by device code memory allocations
    cudaStatus = cudaMalloc(&preAllocatedBufferHost[gpu], PREALLOCATED_MEMORY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s", cudaGetErrorString(cudaStatus));
        exit(0);
    }
    else
    {
        printf("\nAllocated preAllocatedBuffer of %d bytes, address: %X\n", PREALLOCATED_MEMORY_SIZE, preAllocatedBufferHost[gpu]);
    }

    cudaMemset(&preAllocatedMemoryUsed, 0, sizeof(uint32));
}

uint32 estimateLaunchDepth(HexaBitBoardPosition *pos)
{
    // estimate branching factor near the root
    double perft1 = perft_bb(pos, 1);
    double perft2 = perft_bb(pos, 2);
    double perft3 = perft_bb(pos, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
    float geoMean = sqrt((perft3 / perft2) * (perft2 / perft1));
    float arithMean = ((perft3 / perft2) + (perft2 / perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }

    //printf("\nEstimated branching factor: %g\n", branchingFactor);

    float memLimit = PREALLOCATED_MEMORY_SIZE / 2;  // be conservative as the branching factor can increase later

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = log(memLimit) / log(branchingFactor);

    //printf("\nEstimated launch depth: %d\n", depth);

    return depth;
}

int main(int argc, char *argv[])
{
    BoardPosition testBoard;

    cudaGetDeviceCount(&numGPUs);

    printf("No of GPUs detected: %d", numGPUs);

    for (int g = 0; g < numGPUs; g++)
    {
        initGPU(g);
#if USE_TRANSPOSITION_TABLE == 1
        setupHashTables128b(TransTables128b[g]);
#endif
        MoveGeneratorBitboard::init();
    }
    // set default device to device 0
    cudaSetDevice(0);

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
        perftLauncher(&testBB, depth, launchDepth);
    }


    freeHashTables();

    for (int g = 0; g < numGPUs; g++)
    {
        cudaFree(preAllocatedBufferHost[g]);
        cudaDeviceReset();
    }
    
    return 0;
}
