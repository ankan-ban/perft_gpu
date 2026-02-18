#include "device_launch_parameters.h"
#include "perft_bb.h"
#include <math.h>
#include <stdlib.h>

#include "launcher.h"

int main(int argc, char *argv[])
{
    BoardPosition testBoard;

    int totalGPUs;
    cudaGetDeviceCount(&totalGPUs);
    printf("No of GPUs detected: %d\n", totalGPUs);

    if (totalGPUs == 0)
    {
        printf("No CUDA GPUs found. Exiting.\n");
        return 1;
    }

    // Initialize single GPU (device 0)
    initGPU(0);
    MoveGeneratorBitboard::init();

    char fen[1024] = "";
    int maxDepth = 10;

    if (argc >= 3)
    {
        strcpy(fen, argv[1]);
        maxDepth = atoi(argv[2]);
    }
    else
    {
        printf("\nUsage: perft_gpu <fen> <depth> [<launchdepth>]\n");
        printf("\nAs no parameters were provided... running default test\n");
    }

    if (strlen(fen) > 5)
    {
        Utils::readFENString(fen, &testBoard);
    }
    else
    {
        Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard);
    }
    Utils::dispBoard(&testBoard);

    QuadBitBoard testBB;
    GameState testGS;
    Utils::board088ToQuadBB(&testBB, &testGS, &testBoard);

    uint32 launchDepth = estimateLaunchDepth(&testBB, &testGS);
    launchDepth = min(launchDepth, (uint32)11);

    // for best performance without GPU hash
    if (launchDepth < 6)
        launchDepth = 6;

    if (argc >= 4)
    {
        launchDepth = atoi(argv[3]);
    }

    if ((uint32)maxDepth < launchDepth)
    {
        launchDepth = maxDepth;
    }

    printf("Launch depth: %d\n", launchDepth);
    fflush(stdout);

    for (int depth = 1; depth <= maxDepth; depth++)
    {
        perftLauncher(&testBB, &testGS, depth, launchDepth);
        fflush(stdout);
    }

    cudaFree(preAllocatedBufferHost);
    cudaDeviceReset();

    return 0;
}
