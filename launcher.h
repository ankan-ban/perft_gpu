// launcher utils - simplified for single-GPU no-hash perft

void initGPU(int gpu)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! error: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    printf("\ngpu: %d, memory total: %llu, free: %llu\n", gpu, (unsigned long long)total, (unsigned long long)free);

    // allocate the preallocated buffer for BFS tree storage
    cudaStatus = cudaMalloc(&preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    else
    {
        printf("Allocated preAllocatedBuffer of %llu bytes\n", (unsigned long long)PREALLOCATED_MEMORY_SIZE);
    }
}

uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs)
{
    // estimate branching factor near the root
    double perft1 = (double)perft_bb(pos, gs, 1);
    double perft2 = (double)perft_bb(pos, gs, 2);
    double perft3 = (double)perft_bb(pos, gs, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
    float geoMean = sqrt((perft3 / perft2) * (perft2 / perft1));
    float arithMean = ((perft3 / perft2) + (perft2 / perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }

    float memLimit = PREALLOCATED_MEMORY_SIZE / 2;  // be conservative

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = (uint32)(log(memLimit) / log(branchingFactor));

    return depth;
}


// Serial CPU recursion at top levels, launching GPU BFS at launchDepth
uint64 perft_cpu_recurse(QuadBitBoard *pos, GameState *gs, int depth, int launchDepth, void *gpuBuffer, size_t bufferSize)
{
    if (depth <= launchDepth)
    {
        return perft_gpu_host_bfs(pos, gs, depth, gpuBuffer, bufferSize);
    }

    // Serial CPU recursion
    CMove moves[MAX_MOVES];
    QuadBitBoard childPos;
    GameState childGs;
    int nMoves = generateMoves(pos, gs, gs->chance, moves);

    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        childPos = *pos;
        childGs = *gs;
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (gs->chance == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE>(&childPos, &childGs, moves[i]);
        else
            MoveGeneratorBitboard::makeMove<BLACK>(&childPos, &childGs, moves[i]);
#else
        MoveGeneratorBitboard::makeMove(&childPos, &childGs, moves[i], gs->chance);
#endif

        uint64 childPerft = perft_cpu_recurse(&childPos, &childGs, depth - 1, launchDepth, gpuBuffer, bufferSize);
        count += childPerft;
    }
    return count;
}


void perftLauncher(QuadBitBoard *pos, GameState *gs, uint32 depth, int launchDepth)
{
    EventTimer timer;
    timer.start();

    uint64 result = perft_cpu_recurse(pos, gs, depth, launchDepth, preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);

    timer.stop();

    printf("\nPerft(%02d): %llu, time: %g seconds", depth, (unsigned long long)result, timer.elapsed() / 1000.0);
    if (timer.elapsed() > 0)
        printf(", nps: %llu", (unsigned long long)((double)result / timer.elapsed() * 1000.0));
    printf("\n");
    fflush(stdout);
}
