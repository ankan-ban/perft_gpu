// functions for computing perft using bitboard board representation
#include "MoveGeneratorBitboard.h"

#if USE_PREALLOCATED_MEMORY == 1
           void   *preAllocatedBufferHost;
__device__ void   *preAllocatedBuffer;
__device__ uint32  preAllocatedMemoryUsed;
#endif

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
#include "moderngpu-master/include/kernels/scan.cuh"
#include "moderngpu-master/include/kernels/intervalmove.cuh"
#endif

// perft counter function. Returns perft of the given board for given depth
#if USE_MOVE_LIST_FOR_CPU_PERFT == 1
uint64 perft_bb(HexaBitBoardPosition *pos, uint32 depth)
{
    CMove genMoves[MAX_MOVES];
    uint32 nMoves = 0;
    uint8 chance = pos->chance;

#if USE_COUNT_ONLY_OPT == 1
    if (depth == 1)
    {
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (chance == BLACK)
        {
            nMoves = MoveGeneratorBitboard::countMoves<BLACK>(pos);
        }
        else
        {
            nMoves = MoveGeneratorBitboard::countMoves<WHITE>(pos);
        }
#else
        nMoves = MoveGeneratorBitboard::countMoves(pos, chance);
#endif
        return nMoves;
    }
#endif

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        nMoves = MoveGeneratorBitboard::generateMoves<BLACK>(pos, genMoves);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateMoves<WHITE>(pos, genMoves);
    }
#else
    nMoves = MoveGeneratorBitboard::generateMoves(pos, genMoves, chance);
#endif

#if USE_COUNT_ONLY_OPT == 0
    if (depth == 1)
        return nMoves;
#endif


    // Ankan - for testing
    /*
    HexaBitBoardPosition newPositions[MAX_MOVES];
    if (chance == BLACK)
        nMoves = MoveGeneratorBitboard::generateBoards<BLACK>(pos, newPositions);
    else
        nMoves = MoveGeneratorBitboard::generateBoards<WHITE>(pos, newPositions);
    */

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        // copy - make the move
        HexaBitBoardPosition newPos = *pos;
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(&newPos, genMoves[i]);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(&newPos, genMoves[i]);
    }
#else
        MoveGeneratorBitboard::makeMove(&newPos, genMoves[i], chance);
#endif


        // Ankan - for testing
        /*
        if(memcmp(&newPos, &newPositions[i], sizeof(HexaBitBoardPosition)))
        {
            printf("\n\ngot wrong board at index %d", i);
            printf("\nBoard: \n");
            BoardPosition testBoard;
            Utils::boardHexBBTo088(&testBoard, pos);
            Utils::dispBoard(&testBoard);            

            printf("\nMove: ");
            Utils::displayCompactMove(genMoves[i]);

            assert(0);
        }
        */


        uint64 childPerft = perft_bb(&newPos, depth - 1);
        count += childPerft;
    }

    return count;

}
#else
uint64 perft_bb(HexaBitBoardPosition *pos, uint32 depth)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];

#if DEBUG_PRINT_MOVES == 1
    if (depth == DEBUG_PRINT_DEPTH)
        printMoves = true;
    else
        printMoves = false;
#endif    

    uint32 nMoves = 0;
    uint8 chance = pos->chance;

#if USE_COUNT_ONLY_OPT == 1
    if (depth == 1)
    {
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (chance == BLACK)
        {
            nMoves = MoveGeneratorBitboard::countMoves<BLACK>(pos);
        }
        else
        {
            nMoves = MoveGeneratorBitboard::countMoves<WHITE>(pos);
        }
#else
        nMoves = MoveGeneratorBitboard::countMoves(pos, chance);
#endif
        return nMoves;
    }
#endif

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

#if USE_COUNT_ONLY_OPT == 0
    if (depth == 1)
        return nMoves;
#endif

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        uint64 childPerft = perft_bb(&newPositions[i], depth - 1);
#if DEBUG_PRINT_MOVES == 1
        if (depth == DEBUG_PRINT_DEPTH)
            printf("%llu\n", childPerft);
#endif
        count += childPerft;
    }

    return count;
}
#endif


// can be tuned as per need
#define BLOCK_SIZE 256

// fixed
#define WARP_SIZE 32

#define ALIGN_UP(addr, align)   (((addr) + (align) - 1) & (~((align) - 1)))

template<typename T>
__device__ __forceinline__ int deviceMalloc(T **ptr, uint32 size)
{
#if USE_PREALLOCATED_MEMORY == 1
    // align up the size to nearest 4096 bytes
    // There is some bug somewhere that causes problems if the pointer returned is not aligned (or aligned to lesser number)
    // TODO: find the bug and fix it
    size = ALIGN_UP(size, 4096);
    uint32 startOffset = atomicAdd(&preAllocatedMemoryUsed, size);
    if (startOffset >= PREALLOCATED_MEMORY_SIZE)
    {
        // printf("\nFailed allocating %d bytes\n", size);
        return E_FAIL;
    }

    *ptr = (T*) ((uint8 *)preAllocatedBuffer + startOffset);

    //printf("\nAllocated %d bytes at address: %X\n", size, *ptr);

#else
    return cudaMalloc(ptr, size);
#endif

    return S_OK;
}

template<typename T>
__device__ __forceinline__ void deviceFree(T *ptr)
{
#if USE_PREALLOCATED_MEMORY == 1
    // we don't free memory here (memory is freed when the recursive serial kernel gets back the control)
#else
    cudaFree(ptr);
#endif
}

// makes the given move on the given position
__device__ __forceinline__ void makeMove(HexaBitBoardPosition *pos, CMove move, int chance)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(pos, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(pos, move);
    }
#else
    MoveGeneratorBitboard::makeMove(pos, move, chance);
#endif
}

__device__ __forceinline__ uint32 countMoves(HexaBitBoardPosition *pos, uint8 color)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::countMoves<BLACK>(pos);
    }
    else
    {
        return MoveGeneratorBitboard::countMoves<WHITE>(pos);
    }
#else
    return MoveGeneratorBitboard::countMoves(pos, color);
#endif
}

__device__ __forceinline__ uint32 generateBoards(HexaBitBoardPosition *pos, uint8 color, HexaBitBoardPosition *childBoards)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateBoards<BLACK>(pos, childBoards);
    }
    else
    {
        return MoveGeneratorBitboard::generateBoards<WHITE>(pos, childBoards);
    }
#else
    return MoveGeneratorBitboard::generateBoards(pos, childBoards, color);
#endif
}


__device__ __forceinline__ uint32 generateMoves(HexaBitBoardPosition *pos, uint8 color, CMove *genMoves)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateMoves<BLACK>(pos, genMoves);
    }
    else
    {
        return MoveGeneratorBitboard::generateMoves<WHITE>(pos, genMoves);
    }
#else
    return MoveGeneratorBitboard::generateMoves(pos, genMoves, color);
#endif
}

// shared memory scan for entire thread block
__device__ __forceinline__ void scan(uint32 *sharedArray)
{
    uint32 diff = 1;
    while(diff < blockDim.x)
    {
        uint32 val1, val2;
        
        if (threadIdx.x >= diff)
        {
            val1 = sharedArray[threadIdx.x];
            val2 = sharedArray[threadIdx.x - diff];
        }
        __syncthreads();
        if (threadIdx.x >= diff)
        {
            sharedArray[threadIdx.x] = val1 + val2;
        }
        diff *= 2;
        __syncthreads();
    }
}

// fast reduction for the warp
__device__ __forceinline__ void warpReduce(int &x)
{
    #pragma unroll
    for(int mask = 16; mask > 0 ; mask >>= 1)
        x += __shfl_xor(x, mask);
}

// fast scan for the warp
__device__ __forceinline__ void warpScan(int &x, int landId)
{
    #pragma unroll
    for( int offset = 1 ; offset < WARP_SIZE ; offset <<= 1 )
    {
        float y = __shfl_up(x, offset);
        if(landId >= offset)
        x += y;
    }
}

union sharedMemAllocs
{
    struct
    {
        uint32                  movesForThread[BLOCK_SIZE];
        HexaBitBoardPosition    *currentLevelBoards;        // array of size BLOCK_SIZE
        // first level move counts isn't stored anywhere (it's in register 'nMoves')

        // numFirstLevelMoves isn't stored in shared memory
        CMove                   *allFirstLevelChildMoves;   // array of size numFirstLevelMoves
        HexaBitBoardPosition    *allFirstLevelChildBoards;  // array of size numFirstLevelMoves
        uint32                  *allSecondLevelMoveCounts;  // array of size numFirstLevelMoves

        uint32                  numAllSecondLevelMoves;
        CMove                   *allSecondLevelChildMoves;  // array of size numAllSecondLevelMoves
        HexaBitBoardPosition   **boardPointers;             // array of size numAllSecondLevelMoves (second time)
    };
};

__launch_bounds__( BLOCK_SIZE, 4 )
__global__ void perft_bb_gpu_single_level(HexaBitBoardPosition **positions, CMove *moves, uint64 *globalPerftCounter, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        move = moves[index];
        color = pos.chance;
        makeMove(&pos, move, color);
        color = !color;
        nMoves = countMoves(&pos, color);
    }

    // on Kepler, atomics are so fast that one atomic instruction per leaf node is also fast enough (faster than full reduction)!
    // warp-wide reduction seems a little bit faster
    warpReduce(nMoves);

    int laneId = threadIdx.x & 0x1f;

    if (laneId == 0)
    {
        atomicAdd (globalPerftCounter, nMoves);
    }
    return;
}

// this version gets a list of moves, and a list of pointers to BitBoards
// first it makes the move to get the new board and then counts the moves possible on the board
// positions        - array of pointers to old boards
// generatedMoves   - moves to be made
__launch_bounds__( BLOCK_SIZE, 4)
__global__ void makeMove_and_perft_single_level(HexaBitBoardPosition **positions, CMove *generatedMoves, uint64 *globalPerftCounter, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    HexaBitBoardPosition *posPointer = positions[index];
    HexaBitBoardPosition pos = *posPointer;
    int color = pos.chance;

    CMove move = generatedMoves[index];

    makeMove(&pos, move, color);

    // 2. count moves at this position
    int nMoves = 0;
    nMoves = countMoves(&pos, !color);

    // 3. add the count to global counter

    // on Kepler, atomics are so fast that one atomic instruction per leaf node is also fast enough (faster than full reduction)!
    // warp-wide reduction seems a little bit faster
    warpReduce(nMoves);

    int laneId = threadIdx.x & 0x1f;
    
    if (laneId == 0)
    {
        atomicAdd (globalPerftCounter, nMoves);
    }
    
}


// moveCounts are per each thread
// this function first reads input position from *positions[] - which is an array of pointers
// then it makes the given move (moves[] array)
// puts the updated board in outPositions[] array
// and finally counts the no. of moves possible for each element in outPositions.
// the move counts are returned in moveCounts[] array
__launch_bounds__( BLOCK_SIZE, 4 )
__global__ void makemove_and_count_moves_single_level(HexaBitBoardPosition **positions, CMove *moves, HexaBitBoardPosition *outPositions, uint32 *moveCounts, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;

    // just count the no. of moves for each board and save it in moveCounts array
    int nMoves = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        move = moves[index];
        color = pos.chance;
        makeMove(&pos, move, color);
        color = !color;
        outPositions[index] = pos;
        nMoves = countMoves(&pos, color);
    }

    moveCounts[index] = nMoves;
}

// childPositions is array of pointers
__global__ void generate_boards_single_level(HexaBitBoardPosition *positions, HexaBitBoardPosition **childPositions, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos = positions[index];
    HexaBitBoardPosition *childBoards = childPositions[index];

    uint8 color = pos.chance;

    if (index < nThreads)
    {
        generateBoards(&pos, color, childBoards);
    }
}

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN==1
// positions[] array contains positions on which move have to be generated. 
// generatedMovesBase contains the starting address of the memory allocated for storing the generated moves
// moveListIndex points to the start index in the above memory for storing generated moves for current board position
__global__ void generate_moves_single_level(HexaBitBoardPosition *positions, CMove *generatedMovesBase, int *moveListIndex, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos = positions[index];
    CMove *genMoves = generatedMovesBase + moveListIndex[index];

    uint8 color = pos.chance;

    if (index < nThreads)
    {
        generateMoves(&pos, color, genMoves);
    }
}
#else
// positions[] array contains positions on which move have to be generated. 
// generatedMoves is array of pointers, pointing to the memory allocated for each thread to store generated moves
__global__ void generate_moves_single_level(HexaBitBoardPosition *positions, CMove **generatedMoves, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos = positions[index];
    CMove *genMoves = generatedMoves[index];

    uint8 color = pos.chance;

    if (index < nThreads)
    {
        generateMoves(&pos, color, genMoves);
    }
}
#endif

#if 0
// makes the given moves on the given board positions
// no longer used (used only for testing)
__global__ void makeMoves(HexaBitBoardPosition *positions, CMove *generatedMoves, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos = positions[index];

    CMove move = generatedMoves[index];

    // Ankan - for testing
    if (index <= 2)
    {
        Utils::displayCompactMove(move);
    }

    int chance = pos.chance;
    makeMove(&pos, move, chance);
    positions[index] = pos;
}
#endif

// this version launches two levels as a single gird
// to be called only at depth == 3
// ~20 Billion moves per second in best case!
__global__ void perft_bb_gpu_depth3(HexaBitBoardPosition **positions, CMove *moves, uint64 *globalPerftCounter, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        move = moves[index];
        color = pos.chance;
        makeMove(&pos, move, color);
        color = !color;
        nMoves = countMoves(&pos, color);
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 numFirstLevelMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (numFirstLevelMoves == 0)
        return;

    // first thread of block allocates memory to store all moves generated by the thread block
    if (threadIdx.x == 0)
    {
        // TODO: maybe combine these multiple memory requests to a single request (to save on no. of AtmoicAdds() used)
        // allocate memory for:

        // first level child moves
        deviceMalloc(&shMem.allFirstLevelChildMoves, sizeof(CMove) * numFirstLevelMoves);

        // current level board positions
        deviceMalloc(&shMem.currentLevelBoards, sizeof(HexaBitBoardPosition) * BLOCK_SIZE);

        // and also for pointers for first level childs to to current level board positions
        deviceMalloc(&shMem.boardPointers, sizeof(HexaBitBoardPosition *) * numFirstLevelMoves);

        // also need to allocate first level child boards
        deviceMalloc(&shMem.allFirstLevelChildBoards, sizeof(HexaBitBoardPosition) * numFirstLevelMoves);

        // allocate memory to hold counts of moves that will be generated by first level boards
        // first level move counts is in per-thread variable "nMoves"
        int size = sizeof(uint32) * numFirstLevelMoves;
#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
        size = (int) size * 1.2f + 256;
#endif
        // (add some scratch space needed by scan and intervalExpand functions)
        deviceMalloc(&shMem.allSecondLevelMoveCounts, size);
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *currentLevelBoards = shMem.currentLevelBoards;
    currentLevelBoards[threadIdx.x] = pos;

    // make all the board pointers point to the current board
    HexaBitBoardPosition **firstLevelBoardPointers = shMem.boardPointers + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        firstLevelBoardPointers[i] = &currentLevelBoards[threadIdx.x];
    }

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, firstLevelChildMoves);
    }

    __syncthreads();

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
    // use interval expand algorithm taken from http://nvlabs.github.io/moderngpu/intervalmove.html
    // thread 0 of the block does everything from here on (mostly launching new kernels to get work done)
    if (threadIdx.x == 0)
    {
        HexaBitBoardPosition *firstLevelChildBoards = shMem.allFirstLevelChildBoards;
        int *secondLevelMoveCounts = (int *) shMem.allSecondLevelMoveCounts;
        int numSecondLevelMoves = 0;
        cudaStream_t childStream;        
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

        // 1. count the moves that would be generated by childs
        //    (and also generate first level child boards, from child boards)
        uint32 nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;
        makemove_and_count_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, firstLevelChildBoards, (uint32 *) secondLevelMoveCounts, numFirstLevelMoves);

        // 2. secondLevelMoveCounts now has individual move counts, run a scan on it
        int *pNumSecondLevelMoves = secondLevelMoveCounts + numFirstLevelMoves;
        int *scratchSpace = pNumSecondLevelMoves + 1;
        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (secondLevelMoveCounts, numFirstLevelMoves, secondLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	         pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();
        // the scan also gives the total of moveCounts
        numSecondLevelMoves = *pNumSecondLevelMoves;

        if (numSecondLevelMoves == 0)
        {
            cudaStreamDestroy(childStream);
            return;
        }

        // 3. allocate memory for:
        // second level child moves
        CMove *secondLevelChildMoves;
        deviceMalloc(&secondLevelChildMoves, sizeof(CMove) * numSecondLevelMoves);

        // and board pointers to first level boards
        HexaBitBoardPosition **secondLevelBoardPointers;
        deviceMalloc(&secondLevelBoardPointers, sizeof(void *) * numSecondLevelMoves);


        // 4. now run interval expand

        // this call is a bit tricky :
        // we want the pointers of first level child boards to get replicated in the secondLevelBoardPointers array
        // i.e, firstLevelChildBoards + 0  for all childs of the first  'first level child board'
        //      firstLevelChildBoards + 1  for all childs of the second 'first level child board'
        // etc. 
        // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output
        // We typecast the firstLevelChildBoards as integer to get the base number
        // and pass sizeof(HexaBitBoardPosition) as the multiplier
        mgpu::IntervalExpandDGenValues(numSecondLevelMoves, secondLevelMoveCounts, (int) firstLevelChildBoards,     // WARNING: typecasting pointer to integer
                                       sizeof(HexaBitBoardPosition), numFirstLevelMoves, (int *) secondLevelBoardPointers, 
                                       childStream, scratchSpace);

  
        // secondLevelMoveCounts now have the exclusive scan - containing the indices to put moves on

        // 5. generate the second level child moves
        // secondLevelMoveCounts is used by the below kernel to index into secondLevelChildMoves[] - to know where to put the generated moves
        generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelChildBoards, secondLevelChildMoves, secondLevelMoveCounts, numFirstLevelMoves);


        // 6. now we have all second level generated moves in secondLevelChildMoves .. launch a kernel at depth - 2 to make the moves and count leaves
        nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;
        makeMove_and_perft_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>> (secondLevelBoardPointers, secondLevelChildMoves, globalPerftCounter, numSecondLevelMoves);

        // when preallocated memory is used, we don't need to free memory
        // which also means that there is no need to wait for child kernel to finish
#if USE_PREALLOCATED_MEMORY != 1
        cudaDeviceSynchronize();
        cudaStreamDestroy(childStream);
#endif
    }
#else

    HexaBitBoardPosition *firstLevelChildBoards = shMem.allFirstLevelChildBoards + moveListOffset;

    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    cudaStream_t childStream;
    uint32 *secondLevelMoveCounts;
    secondLevelMoveCounts = shMem.allSecondLevelMoveCounts;

    if (threadIdx.x == 0)
    {
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        uint32 nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;

        // count the moves that would be generated by childs
        makemove_and_count_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, shMem.allFirstLevelChildBoards, secondLevelMoveCounts, numFirstLevelMoves);
        cudaDeviceSynchronize();
    }

    __syncthreads();

    // movelistOffset contains starting location of childs for each thread
    uint32 localMoveCounter = 0;    // no. of child moves generated by child boards of this thread
    for (int i = 0; i < nMoves; i++)
    {
        localMoveCounter += secondLevelMoveCounts[moveListOffset + i];
    }

    // put localMoveCounter in shared memory and perform a scan to get first level scan
    shMem.movesForThread[threadIdx.x] = localMoveCounter;

    __syncthreads();
    scan(shMem.movesForThread);

    uint32 numSecondLevelMoves = shMem.movesForThread[blockDim.x - 1];
    if (numSecondLevelMoves == 0)
    {
        if (threadIdx.x == 0)
        {
            cudaStreamDestroy(childStream);
        }
        return;
    }

    // first thread of the block allocates memory for all second level moves and board pointers
    if (threadIdx.x == 0)
    {
        // allocate memory for:
        
        // second level child moves
        deviceMalloc(&shMem.allSecondLevelChildMoves, sizeof(CMove) * numSecondLevelMoves);

        // and board pointers to first level boards
        deviceMalloc(&shMem.boardPointers, sizeof(void *) * numSecondLevelMoves);
    }

    __syncthreads();
    CMove *secondLevelChildMoves = shMem.allSecondLevelChildMoves;

    // do full scan of secondLevelMoveCounts global memory array to get starting offsets of all second level child moves
    // all threads do this in a co-operative way
    uint32 baseOffset = shMem.movesForThread[threadIdx.x] - localMoveCounter;
    HexaBitBoardPosition **boardPointers = shMem.boardPointers;


    // TODO: this operation is expensive
    // fix this by colaesing memory reads/writes

    for (int i = 0; i < nMoves; i++)
    {
        uint32 nChildMoves = secondLevelMoveCounts[moveListOffset + i];
        HexaBitBoardPosition *currentBoardPointer = &firstLevelChildBoards[i];

        for (int j=0; j < nChildMoves; j++)
        {
            // this is about 2000 writes for each thread!
            boardPointers[baseOffset + j] = currentBoardPointer;
        }

        secondLevelMoveCounts[moveListOffset + i] = (uint32) (secondLevelChildMoves + baseOffset);
        baseOffset += nChildMoves;
    }

    __syncthreads();
    // secondLevelMoveCounts now have the exclusive scan - containing the addresses to put moves on

    if (threadIdx.x == 0)
    {
        // first thread of the block now launches kernel that generates second level moves

        uint32 nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;
        generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelChildBoards, (CMove **) secondLevelMoveCounts, numFirstLevelMoves);

        // now we have all second level generated moves in secondLevelChildMoves .. launch a kernel at depth - 2 to make the moves and count leaves
        nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;

        makeMove_and_perft_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>> (boardPointers, secondLevelChildMoves, globalPerftCounter, numSecondLevelMoves);

        // when preallocated memory is used, we don't need to free memory
        // which also means that there is no need to wait for child kernel to finish
#if USE_PREALLOCATED_MEMORY != 1
        cudaDeviceSynchronize();
        cudaStreamDestroy(childStream);
#endif
    }
#endif
}


// similar to above function - but processes 3 levels at a time
// to be called only at depth == 4
// only works with USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN
#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
__global__ void perft_bb_gpu_depth4(HexaBitBoardPosition **positions, CMove *moves, uint64 *globalPerftCounter, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        move = moves[index];
        color = pos.chance;
        makeMove(&pos, move, color);
        color = !color;
        nMoves = countMoves(&pos, color);
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 numFirstLevelMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (numFirstLevelMoves == 0)
        return;

    // first thread of block allocates memory to store all moves generated by the thread block
    if (threadIdx.x == 0)
    {
        // TODO: maybe combine these multiple memory requests to a single request (to save on no. of AtmoicAdds() used)
        // allocate memory for:

        // first level child moves
        deviceMalloc(&shMem.allFirstLevelChildMoves, sizeof(CMove) * numFirstLevelMoves);

        // current level board positions
        deviceMalloc(&shMem.currentLevelBoards, sizeof(HexaBitBoardPosition) * BLOCK_SIZE);

        // and also for pointers for first level childs to to current level board positions
        deviceMalloc(&shMem.boardPointers, sizeof(HexaBitBoardPosition *) * numFirstLevelMoves);

        // also need to allocate first level child boards
        deviceMalloc(&shMem.allFirstLevelChildBoards, sizeof(HexaBitBoardPosition) * numFirstLevelMoves);

        // allocate memory to hold counts of moves that will be generated by first level boards
        // first level move counts is in per-thread variable "nMoves"
        int size = sizeof(uint32) * numFirstLevelMoves;
        size = (int) size * 1.2f + 256;

        // (add some scratch space needed by scan and intervalExpand functions)
        deviceMalloc(&shMem.allSecondLevelMoveCounts, size);
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *currentLevelBoards = shMem.currentLevelBoards;
    currentLevelBoards[threadIdx.x] = pos;

    // make all the board pointers point to the current board
    HexaBitBoardPosition **firstLevelBoardPointers = shMem.boardPointers + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        firstLevelBoardPointers[i] = &currentLevelBoards[threadIdx.x];
    }

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, firstLevelChildMoves);
    }

    __syncthreads();

    // use interval expand algorithm taken from http://nvlabs.github.io/moderngpu/intervalmove.html
    // thread 0 of the block does everything from here on (mostly launching new kernels to get work done)
    if (threadIdx.x == 0)
    {
        HexaBitBoardPosition *firstLevelChildBoards = shMem.allFirstLevelChildBoards;
        int *secondLevelMoveCounts = (int *) shMem.allSecondLevelMoveCounts;
        int numSecondLevelMoves = 0;
        cudaStream_t childStream;        
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

        // 1. count the moves that would be generated by childs
        //    (and also generate first level child boards, from child boards)
        uint32 nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;
        makemove_and_count_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, firstLevelChildBoards, (uint32 *) secondLevelMoveCounts, numFirstLevelMoves);

        // 2. secondLevelMoveCounts now has individual move counts, run a scan on it
        int *pNumSecondLevelMoves = secondLevelMoveCounts + numFirstLevelMoves;
        int *scratchSpace = pNumSecondLevelMoves + 1;
        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (secondLevelMoveCounts, numFirstLevelMoves, secondLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	         pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();
        // the scan also gives the total of moveCounts
        numSecondLevelMoves = *pNumSecondLevelMoves;

        // 3. allocate memory for:
        // second level child moves
        CMove *secondLevelChildMoves;
        deviceMalloc(&secondLevelChildMoves, sizeof(CMove) * numSecondLevelMoves);

        // and board pointers to first level boards
        HexaBitBoardPosition **secondLevelBoardPointers;
        deviceMalloc(&secondLevelBoardPointers, sizeof(void *) * numSecondLevelMoves);


        // 4. now run interval expand

        // this call is a bit tricky :
        // we want the pointers of first level child boards to get replicated in the secondLevelBoardPointers array
        // i.e, firstLevelChildBoards + 0  for all childs of the first  'first level child board'
        //      firstLevelChildBoards + 1  for all childs of the second 'first level child board'
        // etc. 
        // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output
        // We typecast the firstLevelChildBoards as integer to get the base number
        // and pass sizeof(HexaBitBoardPosition) as the multiplier
        mgpu::IntervalExpandDGenValues(numSecondLevelMoves, secondLevelMoveCounts, (int) firstLevelChildBoards,     // WARNING: typecasting pointer to integer
                                       sizeof(HexaBitBoardPosition), numFirstLevelMoves, (int *) secondLevelBoardPointers, 
                                       childStream, scratchSpace);

  
        // secondLevelMoveCounts now have the exclusive scan - containing the indices to put moves on

        // 5. generate the second level child moves
        // secondLevelMoveCounts is used by the below kernel to index into secondLevelChildMoves[] - to know where to put the generated moves
        generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelChildBoards, secondLevelChildMoves, secondLevelMoveCounts, numFirstLevelMoves);


        // 6. now we have all second level generated moves in secondLevelChildMoves
        //  count the number of moves that would be generated by second level childs
        //  (and also generate second level child boards, from first level child board pointers and second level secondLevelChildMoves)

        // firstly allocate space to hold them
        HexaBitBoardPosition *secondLevelChildBoards;
        int *thirdLevelMoveCounts;
        deviceMalloc(&secondLevelChildBoards, sizeof(HexaBitBoardPosition) * numSecondLevelMoves);
        int sizeAlloc = (int) sizeof(uint32) * numSecondLevelMoves * 1.2f + 256;
        deviceMalloc(&thirdLevelMoveCounts, sizeAlloc);

        nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;
        makemove_and_count_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(secondLevelBoardPointers, secondLevelChildMoves, secondLevelChildBoards, (uint32 *) thirdLevelMoveCounts, numSecondLevelMoves);

        // 7. thirdLevelMoveCounts now has individual move counts, run a scan on it
        int *pNumThirdLevelMoves = thirdLevelMoveCounts + numSecondLevelMoves;
        int *scratchSpace2 = pNumThirdLevelMoves + 1;
        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (thirdLevelMoveCounts, numSecondLevelMoves, thirdLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	         pNumThirdLevelMoves, false, childStream, scratchSpace2);

        cudaDeviceSynchronize();
        // the scan also gives the total of moveCounts
        int numThirdLevelMoves = *pNumThirdLevelMoves;

        // 8. allocate memory for:
        // third level child moves
        CMove *thirdLevelChildMoves;
        deviceMalloc(&thirdLevelChildMoves, sizeof(CMove) * numThirdLevelMoves);

        // and board pointers to second level boards
        HexaBitBoardPosition **thirdLevelBoardPointers;
        deviceMalloc(&thirdLevelBoardPointers, sizeof(void *) * numThirdLevelMoves);
        
        // 9. now run interval expand to get thirdLevelBoardPointers filled
        mgpu::IntervalExpandDGenValues(numThirdLevelMoves, thirdLevelMoveCounts, (int) secondLevelChildBoards,     // WARNING: typecasting pointer to integer
                                       sizeof(HexaBitBoardPosition), numSecondLevelMoves, (int *) thirdLevelBoardPointers, 
                                       childStream, scratchSpace2);

        // 10. Generate the third level child moves
        // thirdLevelMoveCounts is used by the below kernel to index into thirdLevelChildMoves[] - to know where to put the generated moves
        generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(secondLevelChildBoards, thirdLevelChildMoves, thirdLevelMoveCounts, numSecondLevelMoves);
        

        // 11. finally run countMoves on third level moves!
        nBlocks = (numThirdLevelMoves - 1) / BLOCK_SIZE + 1;
        makeMove_and_perft_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>> (thirdLevelBoardPointers, thirdLevelChildMoves, globalPerftCounter, numThirdLevelMoves);

        cudaStreamDestroy(childStream);
    }
}
#endif

// this version processes one level a time until it reaches depth 4 - where perft_bb_gpu_depth3 is called
// DON'T CALL this with DEPTH = 1 (call perft_bb_gpu_single_level instead)
// positions - array of pointers to positions on which the given move should be made to get current position
// moves - (array of) the move to make to reach current position
__global__ void perft_bb_gpu_safe(HexaBitBoardPosition **positions,  CMove *moves, uint64 *globalPerftCounter, int depth, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
    {
        if (moves == NULL)
        {
            // really weired hack (sometimes position is array of pointers... and sometimes it's just a pointer!)
            pos = *((HexaBitBoardPosition *)(positions));
            color = pos.chance;
        }
        else
        {
            pos = *(positions[index]);
            move = moves[index];
            color = pos.chance;
            makeMove(&pos, move, color);
            color = !color;
        }

        nMoves = countMoves(&pos, color);
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 numFirstLevelMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (numFirstLevelMoves == 0)
        return;

    if (threadIdx.x == 0)
    {
        // allocate memory for:

        // first level child moves
        deviceMalloc(&shMem.allFirstLevelChildMoves, sizeof(CMove) * numFirstLevelMoves);

        // current level board positions
        deviceMalloc(&shMem.currentLevelBoards, sizeof(HexaBitBoardPosition) * BLOCK_SIZE);

        // and also for pointers for first level childs to to current level board positions
        deviceMalloc(&shMem.boardPointers, sizeof(HexaBitBoardPosition *) * numFirstLevelMoves);
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *boards = shMem.currentLevelBoards;
    boards[threadIdx.x] = pos;

    // make all the board pointers point to the current board
    HexaBitBoardPosition **boardPointers = shMem.boardPointers + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        boardPointers[i] = &boards[threadIdx.x];
    }

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, firstLevelChildMoves);
    }

    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    if (threadIdx.x == 0)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        uint32 nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;

        if (depth == 2)
            perft_bb_gpu_single_level<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, firstLevelChildMoves, globalPerftCounter, numFirstLevelMoves);
#if PARALLEL_LAUNCH_LAST_3_LEVELS == 1
        else if (depth == 5)
            perft_bb_gpu_depth4<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, firstLevelChildMoves, globalPerftCounter, numFirstLevelMoves);
#endif
        else if (depth == 4)
            perft_bb_gpu_depth3<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, firstLevelChildMoves, globalPerftCounter, numFirstLevelMoves);
        else
            perft_bb_gpu_safe<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, firstLevelChildMoves, globalPerftCounter, depth-1, numFirstLevelMoves);

        // when preallocated memory is used, we don't need to free memory
        // which also means that there is no need to wait for child kernel to finish
#if USE_PREALLOCATED_MEMORY != 1
        cudaDeviceSynchronize();
        deviceFree(firstLevelChildMoves);
        deviceFree(boards);
        deviceFree(boardPointers);
#endif
        cudaStreamDestroy(childStream);
    }
}

// traverse the tree recursively (and serially) and launch parallel work on reaching launchDepth
__device__ void perft_bb_gpu_recursive_launcher(HexaBitBoardPosition *pos, uint64 *globalPerftCounter, int depth, HexaBitBoardPosition *boardStack, int launchDepth)
{
    uint32 nMoves = 0;
    uint8 color = pos->chance;

    if (depth == 1)
    {
        nMoves = countMoves(pos, color);
        atomicAdd (globalPerftCounter, nMoves);
    }
    else if (depth <= launchDepth)
    {
        perft_bb_gpu_safe<<<1, BLOCK_SIZE, sizeof(sharedMemAllocs), 0>>> ((HexaBitBoardPosition **) pos, NULL, globalPerftCounter, depth, 1);
        cudaDeviceSynchronize();

        // 'free' up the memory used by the launch
        // printf("\nmemory used by previous parallel launch: %d bytes\n", preAllocatedMemoryUsed);
        preAllocatedMemoryUsed = 0;
    }
    else
    {
        // recurse serially till we reach a depth where we can launch parallel work
        nMoves = generateBoards(pos, color, boardStack);
        //nMoves = generateMoves(pos, color, movesStack);
        for (uint32 i=0; i < nMoves; i++)
        {
            //HexaBitBoardPosition newPos = *pos;
            //makeMove(&newPos, movesStack[i], color);
            perft_bb_gpu_recursive_launcher(&boardStack[i], globalPerftCounter, depth-1, &boardStack[MAX_MOVES], launchDepth);
        }
    }
}

// the starting kernel for perft
__global__ void perft_bb_driver_gpu(HexaBitBoardPosition *pos, uint64 *globalPerftCounter, int depth, HexaBitBoardPosition *boardStack, void *devMemory, int launchDepth)
{
    // set device memory pointer
    preAllocatedBuffer = devMemory;
    preAllocatedMemoryUsed = 0;

    // call the recursive function
    perft_bb_gpu_recursive_launcher(pos, globalPerftCounter, depth, boardStack, launchDepth);
}
