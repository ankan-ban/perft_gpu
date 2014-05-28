#include "perft_bb.h"
#include <math.h>

#define PERFT_VERIF_MODE 0

#if PERFT_VERIF_MODE == 1
#include <time.h>
#endif

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

void initGPU(TTInfo &TTs)
{
    int hr;

#if USE_PREALLOCATED_MEMORY == 1
    // allocate the buffer to be used by device code memory allocations
    hr = cudaMalloc(&preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);
    if (hr != 0)
        printf("error in malloc for preAllocatedBuffer");
    else
        printf("\nAllocated preAllocatedBuffer of %d bytes, address: %X\n", PREALLOCATED_MEMORY_SIZE, preAllocatedBufferHost);

    cudaMemset(&preAllocatedMemoryUsed, 0, sizeof(uint32));
#else
    hr = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1*1024*1024*1024); // 1 GB
    printf("cudaDeviceSetLimit cudaLimitMallocHeapSize returned %d\n", hr);
#endif

#if USE_TRANSPOSITION_TABLE == 1
    // allocate memory for Transposition tables
    setupHashTables(TTs);
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

void removeNewLine(char *str)
{
    while(*str)
    {
        if (*str == '\n' || *str == '\r')
        {
            *str = 0;
            break;
        }
        str++;
    }
}


__global__ void testKernel()
{
    printf("\nHello cuda world!\n");
}


TTInfo TransTables;

// do perft 11 using a single GPU call
// bigger perfts are divided on the CPU
#define GPU_LAUNCH_DEPTH 11

uint64 perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, HexaBitBoardPosition *gpuBoard, uint64 *gpu_perft, void *serial_perft_stack, int launchDepth, char *dispPrefix)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];
    CMove genMoves[MAX_MOVES];
    char  dispString[128];

    uint32 nMoves = 0;

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
                perft_bb_driver_gpu_hash <<<1, 1>>> (gpuBoard, gpu_perft, depth, serial_perft_stack, preAllocatedBufferHost, launchDepth, TransTables);
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

        return res;
    }

    nMoves = generateBoards(pos, newPositions);

    // generate moves also so that we can print them
    generateMoves (pos, pos->chance, genMoves);

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        char moveString[10];
        Utils::getCompactMoveString(genMoves[i], moveString);
        strcpy(dispString, dispPrefix);
        strcat(dispString, moveString);
        uint64 childPerft = perft_bb_cpu_launcher(&newPositions[i], depth - 1, gpuBoard, gpu_perft, serial_perft_stack, launchDepth, dispString);
        printf("%s   %20llu\n", dispString, childPerft);
        fflush(stdout);
        count += childPerft;
    }
    return count;
}


// called only for bigger perfts - shows move count distribution for each move
void dividedPerft(HexaBitBoardPosition *pos, uint32 depth, int launchDepth)
{
    HexaBitBoardPosition *gpuBoard;
    uint64 *gpu_perft;
    void *serial_perft_stack;

    cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
    cudaMalloc(&serial_perft_stack, GPU_SERIAL_PERFT_STACK_SIZE);
    cudaMalloc(&gpu_perft, sizeof(uint64));

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

    initGPU(TransTables);

    MoveGeneratorBitboard::init();

#if PERFT_VERIF_MODE == 1
    FILE *fpInp;    // input file
    FILE *fpOp;     // output file
    int startRecord      = 0;
    int recordsToProcess = 1000000;
    int i = 0;
    if (argc !=5)
    {
        printf("usage: perft14_verif <inFile> <outFile> <startRecord> <recordsToProcess>\n");
        return 0;
    }

    fpInp = fopen(argv[1], "r+");
    fpOp  = fopen(argv[2], "a+");
    startRecord = atoi(argv[3]);
    recordsToProcess = atoi(argv[4]);
    printf("\nStart Record: %d, records to process: %d\n", startRecord, recordsToProcess);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);
    HexaBitBoardPosition *gpuBoard;
    uint64 *gpu_perft;
    HexaBitBoardPosition *serial_perft_stack;
    cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
    cudaMalloc(&serial_perft_stack, GPU_SERIAL_PERFT_STACK_SIZE);
    cudaMalloc(&gpu_perft, sizeof(uint64));

    clock_t start, end;
    start = clock();    
    
    char line[1024];
    int j=0;
    while(fgets(line,1024,fpInp))
    {
        if (j++ < startRecord) continue;

        Utils::readFENString(line, &testBoard);
        HexaBitBoardPosition testBB;

        //Utils::dispBoard(&testBoard);
        printf("\n%s", line);

        Utils::board088ToHexBB(&testBB, &testBoard);
        
        uint32 launchDepth = 6; //estimateLaunchDepth(&testBB);
        //launchDepth = min(launchDepth, 7); // don't go too high

        cudaError_t err = cudaMemcpy(gpuBoard, &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        if (err != S_OK)
            printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));
        cudaMemset(gpu_perft, 0, sizeof(uint64));
#if USE_TRANSPOSITION_TABLE == 1
        perft_bb_driver_gpu_hash <<<1, 1>>> (gpuBoard, gpu_perft, 7, serial_perft_stack, preAllocatedBufferHost, launchDepth, TransTables);
#else
        perft_bb_driver_gpu <<<1, 1>>> (gpuBoard, gpu_perft, 7, serial_perft_stack, preAllocatedBufferHost, launchDepth);
#endif

        uint64 res;
        err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        if (err != S_OK)
            printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

        printf("GPU Perft %d: %llu", 7, res);
        // write to output file
        removeNewLine(line);
        fprintf(fpOp, "%s %llu\n", line, res);
        fflush(fpOp);

        end = clock();
        double t = ((double) end - start) / 1000000;
        printf("\nRecords done: %d, Total: %g seconds, Avg: %g seconds\n", i, t, t / i);
        fflush(stdout);
        i++;
        if (i >= recordsToProcess)
            break;
    }


    cudaFree(gpuBoard);
    cudaFree(gpu_perft);
    cudaFree(serial_perft_stack);

    fclose(fpInp);
    fclose(fpOp);
    return 0;
#endif


    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results
    //Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)
    //Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!
    //Utils::readFENString("r1b1kbnr/pppp1ppp/2n1p3/6q1/6Q1/2N1P3/PPPP1PPP/R1B1KBNR w KQkq - 4 4", &testBoard); // temp test

    int minDepth = 1;
    int maxDepth = 7;
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
        /*
        printf("\nEnter FEN String: \n");
        gets(fen);
        printf("\nEnter max depth: ");
        scanf("%d", &maxDepth);
        */
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

    if (argc >= 4)
    {
        launchDepth = atoi(argv[3]);
    }

    if (maxDepth < launchDepth)
        launchDepth = maxDepth;

    int syncDepth = launchDepth - 1;    // at least last 3 levels are computed by a single kernel launch
#if PARALLEL_LAUNCH_LAST_3_LEVELS == 1
        // sometimes more...
        syncDepth--;
#endif
    if (syncDepth < 2)
        syncDepth = 2;

    // Ankan - for testing
    printf("Calculated syncDepth was: %d\n", syncDepth);
    //syncDepth = 9;

    cudaError_t hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, syncDepth);
    while (hr != S_OK)
    {
        printf("cudaDeviceSetLimit cudaLimitDevRuntimeSyncDepth to depth %d failed. Error: %s ... trying with lower sync depth\n", syncDepth, cudaGetErrorString(hr));
        syncDepth--;
        hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, syncDepth);
    }

    // adjust the launchDepth again in case the cudaDeviceSetLimit call above failed...
    launchDepth = syncDepth + 1;
#if PARALLEL_LAUNCH_LAST_3_LEVELS == 1
    launchDepth++;
#endif

    // Ankan - for testing
    //testSingleLevelPerf(&testBB, 5);
    //testSingleLevelMoveGen(&testBB, 4);
    //maxDepth = 0;

   
    // for testing
    //minDepth = 5;
    //launchDepth = 3;

    launchDepth = 6;    // ankan for testing
    
    for (int depth = minDepth; depth <= maxDepth;depth++)
    {
        dividedPerft(&testBB, depth, launchDepth);
    }
    

#if USE_PREALLOCATED_MEMORY == 1
    cudaFree(preAllocatedBufferHost);
#endif
    cudaDeviceReset();
    return 0;
}
