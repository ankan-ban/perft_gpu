
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
    // 8 should be ok for testing for now
    int hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 6);
    printf("cudaDeviceSetLimit cudaLimitDevRuntimeSyncDepth returned %d\n", hr);

    hr = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1*1024*1024*1024); // 1 GB
    printf("cudaDeviceSetLimit cudaLimitMallocHeapSize returned %d\n", hr);

    /*
    hr = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 256);
    printf("cudaDeviceSetLimit cudaLimitDevRuntimePendingLaunchCount returned %d\n", hr);

    hr = cudaFuncSetCacheConfig(perft_bb_gpu, cudaFuncCachePreferL1);
    printf("cudaFuncSetCacheConfig returned %d\n", hr);


    // we don't really need big stack ?
    hr = cudaDeviceSetLimit(cudaLimitStackSize, 1024);
    printf("cudaDeviceSetLimit stack size returned %d\n", hr);
    */

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
    // Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3

    // no bug till depth 6
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    
    // no bug till depth 6!
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5

    // no bug till depth 7
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!

    //Utils::readFENString("rnb1kb1r/ppqp1ppp/2p5/4P3/2B5/6K1/PPP1N1PP/RNBQ3R b kq - 0 6", &testBoard); // temp test


    /*
    // bug!
    printf("\nsquares between: %llu\n", MoveGeneratorBitboard::squaresInBetween(G8, B3));
    printf("\nsquares between: %llu\n", MoveGeneratorBitboard::squaresInBetween(B3, G8));

    HexaBitBoardPosition newMoves[MAX_MOVES];
    uint32 bbMoves = MoveGeneratorBitboard::generateMoves(&testBB, newMoves);
    */
    
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

    for (int depth=5;depth<7;depth++)
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
        cudaMalloc(&gpu_perft, sizeof(uint64) * (MAX_MOVES + 1));
        cudaError_t err = cudaMemcpy(gpuBoard, &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));

        // gpuBoard[0]            - the board to work on
        // gpuBoard[1...MAXMOVES] - the child boards

        // same with perfts

        EventTimer gputime;
        gputime.start();
        perft_bb_gpu <<<1, 1, 0>>> (gpuBoard, gpu_perft, &gpuBoard[1], &gpu_perft[1], depth);
        gputime.stop();
        printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaDeviceSynchronize();

        uint64 res;
        err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

        printf("\nGPU Perft %d: %llu,   ", depth, res);
        printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) ((res/gputime.elapsed())*1000.0));

        cudaFree(gpuBoard);
#endif
    }
    
    return 0;
}