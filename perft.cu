
//#include "chess.h"
#include "MoveGenerator088.h"
#include "MoveGeneratorBitboard.h"
#include <math.h>



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
#if USE_SCATTER_ALLOC == 1
  //init the heap
  initHeap(512*1024*1024);
#endif
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
        nMoves = MoveGeneratorBitboard::generateBoards<BLACK>(pos, newPositions);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateBoards<WHITE>(pos, newPositions);
    }
#else
    nMoves = MoveGeneratorBitboard::generateBoards(pos, newPositions, chance);
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
    for (int i=0;i<100;i++)
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


void testSingleLevelMoveGen(HexaBitBoardPosition *pos, int depth)
{
    allPos = (HexaBitBoardPosition *) malloc (sizeof(HexaBitBoardPosition) * (4085603+97862));
    printf("\nAddress of CPU memory: %x\n", allPos);
    perft_save_leaves(pos, depth-1);

    printf("\nno. of moves generated: %d\n", posCounter);
    // for testing fully non-divergent perf
    //for (int i=0; i<posCounter;i++)
    //    allPos[i] = *pos;

    HexaBitBoardPosition *gpuBoard = NULL;
    uint64 *gpu_perft = NULL;
    uint32 *moveCounts = NULL;
    cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition) * posCounter);
    //printf("\nAddress of GPU memory: %x\n", gpuBoard);
    cudaMalloc(&gpu_perft, sizeof(uint64));
    cudaMalloc(&moveCounts, sizeof(uint32) * posCounter);
    uint32 *cpuMoveCounts = (uint32 *) malloc(sizeof(uint32) * posCounter);
    cudaError_t err = cudaMemcpy(gpuBoard, allPos, sizeof(HexaBitBoardPosition) * posCounter, cudaMemcpyHostToDevice);
    if (err != S_OK)
        printf("cudaMemcpyHostToDevice returned %s\n", cudaGetErrorString(err));

    cudaMemset(gpu_perft, 0, sizeof(uint64));

    uint32 nBlocks = (posCounter - 1) / BLOCK_SIZE + 1;

    count_moves_single_level <<<nBlocks, BLOCK_SIZE>>> (gpuBoard, moveCounts, posCounter);
    if (cudaGetLastError() != S_OK)
        printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

    err = cudaMemcpy(cpuMoveCounts, moveCounts, sizeof(uint32) * posCounter, cudaMemcpyDeviceToHost);
    printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

    uint32 *cpuMoveCountsBackup = (uint32 *) malloc(sizeof(uint32) * posCounter);
    memcpy(cpuMoveCountsBackup, cpuMoveCounts, sizeof(uint32) * posCounter);

    int nMoves = 0;
    HexaBitBoardPosition *opPos = &allPos[97862];
    for (int i=0;i<posCounter;i++)
    {
        uint32 curCount = cpuMoveCounts[i];
        cpuMoveCounts[i] = nMoves;

        for (int j=0; j<curCount;j++)
        {
            opPos[nMoves+j] = allPos[i];
        }

        nMoves += curCount;
    }

    
    HexaBitBoardPosition *genBoards;
    cudaMalloc(&genBoards, sizeof(HexaBitBoardPosition) * nMoves);
    cudaMemcpy(genBoards, opPos, sizeof(HexaBitBoardPosition) * nMoves, cudaMemcpyHostToDevice);
    /*
    for (int i=0;i<posCounter;i++)
    {
        cpuMoveCounts[i] = (uint32) (genBoards + cpuMoveCounts[i]);
    }
    */
    CMove *genMoves;
    cudaMalloc(&genMoves, sizeof(CMove) * nMoves);


    for (int i=0;i<posCounter;i++)
    {
        cpuMoveCounts[i] = (uint32) (genMoves + cpuMoveCounts[i]);
    }

    err = cudaMemcpy(moveCounts, cpuMoveCounts, sizeof(uint32) * posCounter, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    EventTimer gputime;
    gputime.start();
    for (int i=0;i<100;i++)
        //generate_boards_single_level <<<nBlocks, BLOCK_SIZE, BLOCK_SIZE>>> (gpuBoard, (HexaBitBoardPosition **) moveCounts, posCounter);
          generate_moves_single_level <<<nBlocks, BLOCK_SIZE, BLOCK_SIZE>>> (gpuBoard, (CMove **) moveCounts, posCounter);
    cudaDeviceSynchronize();
    gputime.stop();
    if (cudaGetLastError() != S_OK)
        printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaDeviceSynchronize();

    printf("\nGPU Perft %d: %u,   ", depth, nMoves);
    printf("Time taken to generate moves: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) (((nMoves * 100ull)/gputime.elapsed())*1000.0));

#if 0
    gputime.start();
    nBlocks = (nMoves - 1) / BLOCK_SIZE + 1;
    //for (int i=0;i<100;i++)
        makeMoves <<<nBlocks, BLOCK_SIZE>>>(genBoards, genMoves, nMoves);
    cudaDeviceSynchronize();
    gputime.stop();
    if (cudaGetLastError() != S_OK)
        printf("host side launch for makemove returned: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    printf("Time taken to make moves: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) (((nMoves * 1ull)/gputime.elapsed())*1000.0));

    gputime.start();
    //for (int i=0;i<100;i++)
        perft_bb_gpu_single_level <<<nBlocks, BLOCK_SIZE>>> (genBoards, gpu_perft, nMoves);
    cudaDeviceSynchronize();
    gputime.stop();

    uint64 res;
    err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
    printf("\nGPU Perft %d: %llu,   ", depth + 1, res);
    printf("Time taken to run single level perft/countMoves: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) (((res)/gputime.elapsed())*1000.0));

#endif


#if 1   
    HexaBitBoardPosition **cpuBoardPointers = (HexaBitBoardPosition **) malloc(sizeof(HexaBitBoardPosition *) * nMoves);
    HexaBitBoardPosition **boardPointers;
    cudaMalloc(&boardPointers, sizeof(HexaBitBoardPosition *) * nMoves);

    int g = 0;
    for (int i=0; i < posCounter; i++)
    {
        for (int j=0;j<cpuMoveCountsBackup[i];j++)
        {
            //if (g <= 2)
            //    printf("board %d is: %X", g, &gpuBoard[i]);

            cpuBoardPointers[g++] = &gpuBoard[i];
        }
    }
    assert(g == nMoves);
    
    cudaMemcpy(boardPointers, cpuBoardPointers, sizeof(HexaBitBoardPosition *) * nMoves, cudaMemcpyHostToDevice);

    cudaMemset(gpu_perft, 0, sizeof(uint64));

    nBlocks = (nMoves - 1) / BLOCK_SIZE + 1;
    gputime.start();
    for (int i=0;i<100;i++)
        makeMove_and_perft_single_level <<<nBlocks, BLOCK_SIZE>>>(boardPointers, genMoves, gpu_perft, nMoves);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != S_OK)
        printf("host side launch for makemove returned: %s\n", cudaGetErrorString(cudaGetLastError()));

    uint64 res=0;
    err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
    printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));

    gputime.stop();
    printf("\nGPU Perft %d: %llu,   ", depth + 1, res);
    printf("Time taken to run make move + single level perft/countMoves: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) (((res)/gputime.elapsed())*1000.0));
    

    free (cpuBoardPointers);
    free (cpuMoveCountsBackup);
    cudaFree(boardPointers);
#endif
    free (cpuMoveCounts);
    cudaFree(gpuBoard);
    cudaFree(genBoards);
    cudaFree(genMoves);
    cudaFree(gpu_perft);
    cudaFree(moveCounts);
    free(allPos);
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
        
    printf("\nEstimated branching factor: %g\n", branchingFactor);

    float memLimit = PREALLOCATED_MEMORY_SIZE / 2;  // be conservative as the branching factor can increase later

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = log(memLimit) / log (branchingFactor);

    printf("\nEstimated launch depth: %d\n", depth);

    return depth;
}

int main()
{
    BoardPosition testBoard;

    initGPU();
    MoveGeneratorBitboard::init();

    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results

    // no bug bug till depth 7
    Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions

    // No bug till depth 6!
    //Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)

    // No bug till depth 7!
    // Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3

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

    // launchDepth is the depth at which the driver kernel launches the work kernels
    // we decide launch depth based by estimating memory requirment of the work kernel that would be launched.
    uint32 launchDepth = estimateLaunchDepth(&testBB);

    int minDepth = 1;
    int maxDepth = 10;

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
        
#if TEST_GPU_PERFT == 1
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
        //perft_bb_gpu <<<1, 1, BLOCK_SIZE * sizeof(uint32)>>> (gpuBoard, gpu_perft, depth, 1);
        //for(int i=0;i<100;i++)
            //perft_bb_gpu_safe <<<1, 1, BLOCK_SIZE * sizeof(uint32)>>> (gpuBoard, gpu_perft, depth, 1);
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
#endif
    }

#if USE_PREALLOCATED_MEMORY == 1
    cudaFree(preAllocatedBufferHost);
#endif
    cudaDeviceReset();
    return 0;
}