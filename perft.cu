
//#include "chess.h"
//#include "MoveGenerator088.h"
#include "MoveGeneratorBitboard.h"
#include <math.h>

#define PERFT_VERIF_MODE 0

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
    LARGE_INTEGER count1, count2, freq; \
    QueryPerformanceFrequency (&freq);  \
    QueryPerformanceCounter(&count1);

#define STOP_TIMER \
    QueryPerformanceCounter(&count2); \
    gTime = ((double)(count2.QuadPart-count1.QuadPart)*1000.0)/freq.QuadPart; \
    }
// for timing CPU code : end

void initGPU()
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

int main(int argc, char *argv[])
{
    BoardPosition testBoard;

    initGPU();
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

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 7);
    HexaBitBoardPosition *gpuBoard;
    uint64 *gpu_perft;
    HexaBitBoardPosition *serial_perft_stack;
    cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
    cudaMalloc(&serial_perft_stack, sizeof(HexaBitBoardPosition) * MAX_GAME_LENGTH * MAX_MOVES);
    cudaMalloc(&gpu_perft, sizeof(uint64));

    LARGE_INTEGER count1, count2, time, freq;
    QueryPerformanceCounter(&count1);
    QueryPerformanceFrequency (&freq);

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
        uint32 launchDepth = estimateLaunchDepth(&testBB);
        launchDepth = min(launchDepth, 7); // don't go too high

        cudaError_t err = cudaMemcpy(gpuBoard, &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        if (err != S_OK)
            printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));
        cudaMemset(gpu_perft, 0, sizeof(uint64));
        perft_bb_driver_gpu <<<1, 1>>> (gpuBoard, gpu_perft, 7, serial_perft_stack, preAllocatedBufferHost, launchDepth);

        uint64 res;
        err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        if (err != S_OK)
            printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

        printf("GPU Perft %d: %llu", 7, res);
        // write to output file
        removeNewLine(line);
        fprintf(fpOp, "%s %llu\n", line, res);
        fflush(fpOp);

        QueryPerformanceCounter(&count2);
        time.QuadPart = (count2.QuadPart - count1.QuadPart);    
        double t = ((double) time.QuadPart) / freq.QuadPart;
        printf("\nRecords done: %d, Total: %g seconds, Avg: %g seconds\n", i, t, t / i);

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
    Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    //Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)
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
        printf("\nEnter FEN String: \n");
        gets(fen);
        printf("\nEnter max depth: ");
        scanf("%d", &maxDepth);
    }

    if (strlen(fen) > 5)
    {
        Utils::readFENString(fen, &testBoard);
    }
    Utils::dispBoard(&testBoard);

    

    HexaBitBoardPosition testBB;
    Utils::board088ToHexBB(&testBB, &testBoard);
    Utils::boardHexBBTo088(&testBoard, &testBB);

    // launchDepth is the depth at which the driver kernel launches the work kernels
    // we decide launch depth based by estimating memory requirment of the work kernel that would be launched.
    uint32 launchDepth = estimateLaunchDepth(&testBB);
    launchDepth = min(launchDepth, 11); // don't go too high

    if (argc >= 4)
    {
        launchDepth = atoi(argv[3]);
    }

    if (maxDepth < launchDepth)
        launchDepth = maxDepth;

    int hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, launchDepth);
    if (hr != S_OK)
        printf("cudaDeviceSetLimit cudaLimitDevRuntimeSyncDepth returned %d\n", hr);

    
    // Ankan - for testing
    //testSingleLevelPerf(&testBB, 5);
    //testSingleLevelMoveGen(&testBB, 4);
    //maxDepth = 0;

    
    for (int depth = minDepth; depth <= maxDepth;depth++)
    {
        /*
        START_TIMER
        bbMoves = perft_bb(&testBB, depth);
        STOP_TIMER
        printf("\nPerft %d: %llu,   ", depth, bbMoves);
        printf("Time taken: %g seconds, nps: %llu\n", gTime/1000.0, (uint64) ((bbMoves/gTime)*1000.0));
        */
        
        // try the same thing on GPU
        HexaBitBoardPosition *gpuBoard;
        uint64 *gpu_perft;
        HexaBitBoardPosition *serial_perft_stack;

        cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
        cudaMalloc(&serial_perft_stack, sizeof(HexaBitBoardPosition) * MAX_GAME_LENGTH * MAX_MOVES);
        cudaMalloc(&gpu_perft, sizeof(uint64));
        cudaError_t err = cudaMemcpy(gpuBoard, &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        if (err != S_OK)
            printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));

        cudaMemset(gpu_perft, 0, sizeof(uint64));

        // gpu_perft is a single 64 bit integer which is updated using atomic adds by leave nodes

        EventTimer gputime;
        gputime.start();
        perft_bb_driver_gpu <<<1, 1>>> (gpuBoard, gpu_perft, depth, serial_perft_stack, preAllocatedBufferHost, launchDepth);
        gputime.stop();
        if (cudaGetLastError() < 0)
            printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaDeviceSynchronize();

        uint64 res;
        err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        if (err != S_OK)
            printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

        printf("\nGPU Perft %d: %llu,   ", depth, res);
        printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) ((res/gputime.elapsed())*1000.0));

        cudaFree(gpuBoard);
        cudaFree(gpu_perft);
        cudaFree(serial_perft_stack);
    }

#if USE_PREALLOCATED_MEMORY == 1
    cudaFree(preAllocatedBufferHost);
#endif
    cudaDeviceReset();
    return 0;
}