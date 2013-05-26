
//#include "chess.h"
#include "MoveGenerator088.h"
#include "MoveGeneratorBitboard.h"






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
    int hr = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1*1024*1024*1024); // 1 GB
    printf("cudaDeviceSetLimit cudaLimitMallocHeapSize returned %d\n", hr);

#if USE_SCATTER_ALLOC == 1
  //init the heap
  initHeap(512*1024*1024);
#endif
    /*        
    hr = cudaDeviceSetLimit(cudaLimitStackSize, 512);
    printf("cudaDeviceSetLimit stack size returned %d\n", hr);

    hr = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32);
    printf("cudaDeviceSetLimit cudaLimitDevRuntimePendingLaunchCount returned %d\n", hr);

    hr = cudaFuncSetCacheConfig(perft_bb_gpu, cudaFuncCachePreferL1);
    printf("cudaFuncSetCacheConfig returned %d\n", hr);


    // we don't really need big stack ?
    */

}

// this version saves all the leaf nodes in memory
HexaBitBoardPosition *allPos;
uint32 posCounter = 0;

uint64 perft_save_leaves(HexaBitBoardPosition *pos, uint32 depth)
{
    HexaBitBoardPosition localArray[256];
    HexaBitBoardPosition *newPositions = localArray;

    uint32 nMoves = 0;
    uint8 chance = pos->chance;

    if (depth == 1)
    {
        newPositions = &allPos[posCounter];
    }

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        nMoves = MoveGeneratorBitboard::generateMoves<BLACK, false, false>(pos, newPositions, 1);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateMoves<WHITE, false, false>(pos, newPositions, 1);
    }
#else
    nMoves = MoveGeneratorBitboard::generateMoves(pos, newPositions, chance, false, false, depth);
#endif

    if (depth == 1)
    {
        posCounter += nMoves;
        return nMoves;
    }

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        uint64 childPerft = perft_save_leaves(&newPositions[i], depth - 1);
        count += childPerft;
    }

    return count;
}

void testSingleLevelPerf(HexaBitBoardPosition *pos, int depth)
{
    allPos = (HexaBitBoardPosition *) malloc (sizeof(HexaBitBoardPosition) * 24376626);
    printf("\nAddress of CPU memory: %x\n", allPos);
    perft_save_leaves(pos, depth-1);

    printf("\nno. of moves generated: %d\n", posCounter);
    // for testing fully non-divergent perf
    //for (int i=0; i<posCounter;i++)
    //    allPos[i] = *pos;

    HexaBitBoardPosition *gpuBoard = NULL;
    uint64 *gpu_perft = NULL;

    cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition) * posCounter);
    printf("\nAddress of GPU memory: %x\n", gpuBoard);
    cudaMalloc(&gpu_perft, sizeof(uint64));
    cudaError_t err = cudaMemcpy(gpuBoard, allPos, sizeof(HexaBitBoardPosition) * posCounter, cudaMemcpyHostToDevice);
    if (err != S_OK)
        printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));

    cudaMemset(gpu_perft, 0, sizeof(uint64));

    uint32 nBlocks = (posCounter - 1) / BLOCK_SIZE + 1;

    EventTimer gputime;
    gputime.start();
    for (int i=0;i<10;i++)
        perft_bb_gpu_single_level <<<nBlocks, BLOCK_SIZE>>> (gpuBoard, gpu_perft, posCounter);
        //perft_bb_gpu <<<nBlocks, BLOCK_SIZE, sizeof(uint32) * BLOCK_SIZE>>> (gpuBoard, gpu_perft, 1, posCounter);
    gputime.stop();
    if (cudaGetLastError() != S_OK)
        printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaDeviceSynchronize();

    uint64 res;
    err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
    printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

    printf("\nGPU Perft %d: %llu,   ", depth, res);
    printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) ((res/gputime.elapsed())*1000.0));

    cudaFree(gpuBoard);
    cudaFree(gpu_perft);
    free(allPos);
}

int main()
{
    BoardPosition testBoard;

    initGPU();
    MoveGeneratorBitboard::init();

    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results

    // no bug bug till depth 7
    //Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions

    // No bug till depth 6!
    Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)

    // No bug till depth 7!
    //Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3

    // no bug till depth 6
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    
    // no bug till depth 6!
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5

    // no bug till depth 7
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!

    //Utils::readFENString("rnb1kbnr/pp1pp1pp/1q3p2/2p5/3P4/N4P2/PPP1P1PP/R1BQKBNR w KQkq - 2 4", &testBoard); // temp test

    
    /*
    printf("\nEnter FEN String: \n");
    char fen[1024];
    gets(fen);
    Utils::readFENString(fen, &testBoard); // start.. 20 positions
    Utils::dispBoard(&testBoard);
    */

    HexaBitBoardPosition testBB;
    Utils::board088ToHexBB(&testBB, &testBoard);
    Utils::boardHexBBTo088(&testBoard, &testBB);


    uint64 bbMoves;

    int maxDepth = 8;

    int hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, maxDepth-1);
    if (hr != S_OK)
        printf("cudaDeviceSetLimit cudaLimitDevRuntimeSyncDepth returned %d\n", hr);

    
    // Ankan - for testing
    //testSingleLevelPerf(&testBB, 8);
    //maxDepth = 0;


    for (int depth=2;depth <= maxDepth;depth++)
    {
        /*
        START_TIMER
        bbMoves = perft_bb(&testBB, depth);
        STOP_TIMER
        printf("\nPerft %d: %llu,   ", depth, bbMoves);
        printf("Time taken: %g seconds, nps: %llu\n", gTime/1000.0, (uint64) ((bbMoves/gTime)*1000.0));
        */
        
#if TEST_GPU_PERFT == 1
        // try the same thing on GPU
        HexaBitBoardPosition *gpuBoard;
        uint64 *gpu_perft;
        cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition) * (MAX_MOVES + 1));
        cudaMalloc(&gpu_perft, sizeof(uint64));
        cudaError_t err = cudaMemcpy(gpuBoard, &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        if (err != S_OK)
            printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));

        cudaMemset(gpu_perft, 0, sizeof(uint64));

        // gpuBoard[0]            - the board to work on
        // gpuBoard[1...MAXMOVES] - the child boards

        // gpu_perft is a single 64 bit integer which is updated using atomic adds by leave nodes

        EventTimer gputime;
        gputime.start();
        perft_bb_gpu <<<1, 1, BLOCK_SIZE * sizeof(uint32)>>> (gpuBoard, gpu_perft, depth, 1);
        gputime.stop();
        if (cudaGetLastError() != S_OK)
            printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaDeviceSynchronize();

        uint64 res;
        err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

        printf("\nGPU Perft %d: %llu,   ", depth, res);
        printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) ((res/gputime.elapsed())*1000.0));

        cudaFree(gpuBoard);
        //cudaFree(gpu_perft);
#endif
    }
    
    return 0;
}