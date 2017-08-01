#include "device_launch_parameters.h"
#include "perft_bb.h"
#include <math.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include "InfInt.h"

#include "launcher.h"

void createNetworkThread();
void endNetworkThread();

int main(int argc, char *argv[])
{
    std::srand(std::time(0));
#if PERFT_RECORDS_MODE == 1
    processPerftRecords(argc, argv);
    return 0;
#endif

    BoardPosition testBoard;

    int totalGPUs;
    cudaGetDeviceCount(&totalGPUs);

    printf("No of GPUs detected: %d", totalGPUs);

    if (argc >= 4)
    {
        numGPUs = atoi(argv[3]);
        if (numGPUs < 1) 
            numGPUs = 1;
        if (numGPUs > totalGPUs) 
            numGPUs = totalGPUs;
        printf("\nUsing %d GPUs\n", numGPUs);
    }
    else
    {
        numGPUs = totalGPUs;
    }

    checkAndCreateDiskHash();

    allocCompleteTT();

#if MULTI_NODE_NETWORK_MODE == 1
    createNetworkThread();
#endif    

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

#if USE_TRANSPOSITION_TABLE == 0
    // for best performance without GPU hash (also set PREALLOCATED_MEMORY_SIZE to 3 x 768MB)
    launchDepth = 6;    // ankan - test!
#endif

    if (argc >= 5)
    {
        launchDepth = atoi(argv[4]);
    }

    if (maxDepth < launchDepth)
    {
        launchDepth = maxDepth;
    }

    for (int depth = minDepth; depth <= maxDepth; depth++)
    {
        perftLauncher(&testBB, depth, launchDepth);
    }

#if MULTI_NODE_NETWORK_MODE == 1
    endNetworkThread();
#endif        

#if USE_TRANSPOSITION_TABLE == 1
    freeHashTables();
#endif

    freeCompleteTT();

    for (int g = 0; g < numGPUs; g++)
    {
        cudaFree(preAllocatedBufferHost[g]);
        cudaDeviceReset();
    }

#if USE_TRANSPOSITION_TABLE == 1    
    printf("\nComplete hash sysmem memory usage: %llu bytes\n", ((uint64) chainIndex) * sizeof(CompleteHashEntry));
    printf("\nMax tree storage GPU memory usage: %llu bytes\n", maxMemoryUsage);
    printf("Regular depth %d Launches: %d\n", GPU_LAUNCH_DEPTH, numRegularLaunches);
    printf("Retry launches: %d\n", numRetryLaunches);
    printf("No of work items recieved from peers: %llu\n", numItemsFromPeers);
#endif

    return 0;
}
