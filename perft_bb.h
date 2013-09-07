// functions for computing perft using bitboard board representation

// the routines that actually generate the moves
#include "MoveGeneratorBitboard.h"

//#if USE_PREALLOCATED_MEMORY == 1
           void   *preAllocatedBufferHost;
__device__ void   *preAllocatedBuffer;
__device__ uint32  preAllocatedMemoryUsed;
//#endif

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
#include "moderngpu-master/include/kernels/scan.cuh"
#include "moderngpu-master/include/kernels/intervalmove.cuh"
#endif


#if COUNT_NUM_COUNT_MOVES == 1
__device__ uint64 numCountMoves;
#endif

#if PRINT_HASH_STATS == 1
// stats for each depth
// numProbes - no. of times hash table was probed (looked up)
// numHits   - no. of times we got what we wanted
// numWrites - no. of times an entry was written or updated in hash table
__device__ uint64  numProbes[MAX_GAME_LENGTH];
__device__ uint64  numHits[MAX_GAME_LENGTH];
__device__ uint64  numStores[MAX_GAME_LENGTH];
#endif

// helper routines for CPU perft
uint32 countMoves(HexaBitBoardPosition *pos)
{
    uint32 nMoves;
    int chance = pos->chance;

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

uint32 generateBoards(HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions)
{
    uint32 nMoves;
    int chance = pos->chance;
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
   
    return nMoves;
}



// A very simple CPU routine - only for estimating launch depth
// this version doesn't use incremental hash
uint64 perft_bb(HexaBitBoardPosition *pos, uint32 depth)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];

    uint32 nMoves = 0;

    if (depth == 1)
    {
        nMoves = countMoves(pos);
        return nMoves;
    }

    nMoves = generateBoards(pos, newPositions);

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        uint64 childPerft = perft_bb(&newPositions[i], depth - 1);
        count += childPerft;
    }
    return count;
}




// can be tuned as per need
#define BLOCK_SIZE 256

// fixed
#define WARP_SIZE 32

#define ALIGN_UP(addr, align)   (((addr) + (align) - 1) & (~((align) - 1)))
#define MEM_ALIGNMENT 16
template<typename T>
__device__ __forceinline__ int deviceMalloc(T **ptr, uint32 size)
{
#if USE_PREALLOCATED_MEMORY == 1
    // align up the size to nearest 16 bytes (as some structures might have assumed 16 byte alignment?)
    size = ALIGN_UP(size, MEM_ALIGNMENT);
    uint32 startOffset = atomicAdd(&preAllocatedMemoryUsed, size);

    /*
    if (startOffset >= PREALLOCATED_MEMORY_SIZE)
    {
        // printf("\nFailed allocating %d bytes\n", size);
        return -1;
    }
    */

    *ptr = (T*) ((uint8 *)preAllocatedBuffer + startOffset);

    //printf("\nAllocated %d bytes at address: %X\n", size, *ptr);

#else
    return cudaMalloc(ptr, size);
#endif

    return S_OK;
}

// fulfill multiple memory requests using a single deviceMalloc
__device__ void deviceMultiAlloc(void **pointers[], int sizes[], int n)
{
    // perform a serial scan to figure out running sum and total
    int mallocSize = 0;
    for (int i=0; i < n; i++)
    {
        int size = ALIGN_UP(sizes[i], MEM_ALIGNMENT);
        sizes[i] = mallocSize;
        mallocSize += size;
    }

    // call deviceMalloc only once to allocate all the needed memory
    uint8 *memory;
    deviceMalloc(&memory, mallocSize);

    // set the pointers with the allocated memory
    for (int i=0; i < n; i++)
    {
        *(pointers[i]) = memory + sizes[i];
    }
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
    uint64 unused;
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK, false>(pos, unused, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE, false>(pos, unused, move);
    }
#else
    MoveGeneratorBitboard::makeMove(pos, unused, move, chance, false);
#endif
}

// this one also updates the hash
__device__ __forceinline__ uint64 makeMoveAndUpdateHash(HexaBitBoardPosition *pos, uint64 hash, CMove move, int chance)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK, true>(pos, hash, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE, true>(pos, hash, move);
    }
#else
    MoveGeneratorBitboard::makeMove(pos, hash, move, chance, true);
#endif

    return hash;
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

struct TTInfo
{
    // shallow depth tables (the perft value is assumed to fit in index bits)
    uint64 *depth2;    // shallow transposition table for depth 2
    uint64 *depth3;    // .. for depth 3
    uint64 *depth4;    // .. for depth 4
    uint64 *depth5;    // .. for depth 5 -> could be risky to use

    // deep transposition tables (could be shared among multiple depths)
    TT_Entry *deepTT[16]; // pointers to transposition table for every depth

    // mask of index and hash bits for each transposition table
    uint64 indexBits[16];
    uint64 hashBits [16];
};

union sharedMemAllocs
{
    struct
    {
        // scratch space of 1 element per thread used to perform thread-block wide operations
        // (mostly scans)
        uint32                  movesForThread[BLOCK_SIZE];

        // pointers to various arrays allocated in device memory


        HexaBitBoardPosition    *currentLevelBoards;        // [BLOCK_SIZE]
        uint32                  *perft4Counters;            // [BLOCK_SIZE], only used by depth4 kernel
        union
        {
            uint64              *perftCounters;             // [BLOCK_SIZE], only used by the main kernel
            uint32              *perft3Counters;            // [BLOCK_SIZE] when used by the depth3 kernel
                                                            // and [numFirstLevelMoves] when used by depth4 kernel
        };

        uint64                  *currentLevelHashes;        // [BLOCK_SIZE]

        // first level move counts isn't stored anywhere (it's in register 'nMoves')

        // numFirstLevelMoves isn't stored in shared memory
        CMove                   *allFirstLevelChildMoves;   // [numFirstLevelMoves]
        HexaBitBoardPosition    *allFirstLevelChildBoards;  // [numFirstLevelMoves]
        uint32                  *allSecondLevelMoveCounts;  // [numFirstLevelMoves]
        uint64                 **counterPointers;           // [numFirstLevelMoves] (only used by main kernel)
        uint64                  *firstLevelHashes;          // [numFirstLevelMoves]
        int                     *firstToCurrentLevelIndices;// [numFirstLevelMoves] used instead of boardpointers in the new depth3 hash kernel
        uint32                  *perft2Counters;            // [numFirstLevelMoves] when used in the depth3 hash kernel 
                                                            // and [numSecondLevelMoves] when used by depth4 hash kernel

        uint32                  numAllSecondLevelMoves;
        CMove                   *allSecondLevelChildMoves;  // [numAllSecondLevelMoves]
        HexaBitBoardPosition   **boardPointers;             // [numAllSecondLevelMoves] (second time)
        int                     *secondToFirstLevelIndices; // [numAllSecondLevelMoves] used instead of boardpointers in the new depth3 hash kernel
    };
};

#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4 )
#endif
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
#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4)
#endif
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

// this version gets seperate perft counter per thread
// perftCounters[] is array of pointers to perft counters - where each thread should atomically add the computed perft
#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4)
#endif
__global__ void makeMove_and_perft_single_level(HexaBitBoardPosition **positions, CMove *generatedMoves, uint64 **perftCounters, int nThreads)
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
    uint64 *perftCounter = perftCounters[index];

    // basically check if all threads in the warp are going to atomic add to the same counter, 
    // and if so perform warpReduce and do a single atomic add

    // last 32 bits of counter pointer
#if 1
    int counterIndex = (int) (((uint64) perftCounter) & 0xFFFFFFFF);
    int firstLaneCounter = __shfl(counterIndex, 0);

    if (__all(firstLaneCounter == counterIndex))
    {
        warpReduce(nMoves);

        int laneId = threadIdx.x & 0x1f;
        
        if (laneId == 0)
        {
            atomicAdd (perftCounter, nMoves);
        }
    }
    else
#endif
    {
        atomicAdd (perftCounter, nMoves);
    }
}


// this version uses the indices[] array to index into parentPositions[] and parentCounters[] arrays
#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4)
#endif
__global__ void makeMove_and_perft_single_level_indices(HexaBitBoardPosition *parentBoards, uint32 *parentCounters, 
                                                        int *indices, CMove *moves, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    int parentIndex = indices[index];
    HexaBitBoardPosition pos = parentBoards[parentIndex];
    int color = pos.chance;

    CMove move = moves[index];

    makeMove(&pos, move, color);

    // 2. count moves at this position
    int nMoves = 0;
    nMoves = countMoves(&pos, !color);

    // 3. add the count to global counter

    uint32 *perftCounter = parentCounters + parentIndex;

    // basically check if all threads in the warp are going to atomic add to the same counter, 
    // and if so perform warpReduce and do a single atomic add
    int firstLaneIndex = __shfl(parentIndex, 0);
    if (__all(firstLaneIndex == parentIndex))
    {
        warpReduce(nMoves);

        int laneId = threadIdx.x & 0x1f;
        
        if (laneId == 0)
        {
            atomicAdd (perftCounter, nMoves);
        }
    }
    else
    {
        atomicAdd (perftCounter, nMoves);
    }
}



// moveCounts are per each thread
// this function first reads input position from *positions[] - which is an array of pointers
// then it makes the given move (moves[] array)
// puts the updated board in outPositions[] array
// and finally counts the no. of moves possible for each element in outPositions.
// the move counts are returned in moveCounts[] array
template <bool genBoard>
#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4 )
#endif
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
        if (genBoard)
            outPositions[index] = pos;
        nMoves = countMoves(&pos, color);
    }

    moveCounts[index] = nMoves;
}

// this kernel does several things
// 1. Figures out the parent board position using indices[] array to lookup in parentBoards[] array
// 2. makes the move on parent board to produce current board. Writes it to outPositions[], also updates outHashes with new hash
// 3. looks up the transposition table to see if the current board is present, and if so, updates the perftCounter directly
// 4. which perft counter to update and the hash of parent board is also found by 
//    indexing using indices[] array into parentHashes[]/parentCounters[] arrays
// 5. clears the perftCountersCurrentDepth[] array passed in
#if LIMIT_REGISTER_USE == 1
__launch_bounds__( BLOCK_SIZE, 4 )
#endif
__global__ void makemove_and_count_moves_single_level_hash(HexaBitBoardPosition *parentBoards, uint64 *parentHashes, 
                                                           uint32 *parentCounters, int *indices,  CMove *moves, 
                                                           uint64 *hashTable, uint64 hashBits, uint64 indexBits,
                                                           HexaBitBoardPosition *outPositions, uint64 *outHashes,
                                                           uint32 *moveCounts, uint32 *perftCountersCurrentDepth, 
                                                           int nThreads, int depth)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    // count the no. of moves for each board and save it in moveCounts array
    int nMoves = 0;

    if (index < nThreads)
    {
        int parentIndex = indices[index];
        HexaBitBoardPosition pos = parentBoards[parentIndex];
        uint64 hash = parentHashes[parentIndex];
        uint32 *perftCounter = parentCounters + parentIndex;
        CMove move = moves[index];

        uint8 color = pos.chance;
        hash = makeMoveAndUpdateHash(&pos, hash, move, color);

#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[depth], 1);
#endif
        // check in transposition table
        uint64 entry = hashTable[hash & indexBits];
        if ((entry & hashBits) == (hash & hashBits))
        {
            // hash hit
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[depth], 1);
#endif
            atomicAdd(perftCounter, entry & indexBits);
            pos.whitePieces = 0;    // mark it invalid so that generatemoves doesn't generate moves
            hash = 0;               // mark it invalid so that perft3 to perft2 kernel doesn't process this
            //nMoves = countMoves(&pos, !color);
        }
        else
        {
            nMoves = countMoves(&pos, !color);
        }

        outPositions[index] = pos;
        outHashes[index] = hash;
        moveCounts[index] = nMoves;
        perftCountersCurrentDepth[index] = 0;
    }
}


// compute perft N from perft N-1 (using excessive atomic adds)
// also store perft (N-1) entry in the given hash table (hashes[] array is for positions at N-1 level)
// 'depth' is the value of (n-1)
__global__ void calcPerftNFromPerftNminus1(uint32 *perftNCounters, int *indices, 
                                           uint32 *perftNminus1Counters, uint64 *hashes,
                                           uint64 *hashTable, uint64 hashBits, uint64 indexBits,
                                           int nThreads, int depth)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nThreads)
    {
        uint64 hash = hashes[index];

        if (hash)  // hash == 0 means invalid entry - entry for which there was a hash hit
        {
            uint32 *perftNCounter = perftNCounters + indices[index];
            uint32 perftNminus1 = perftNminus1Counters[index];

            // get perft3 from perft2
            // TODO: try replacing this atomic add with some parallel reduction trick (warp wide?)
            atomicAdd(perftNCounter, perftNminus1);

            // store in hash table
            // it's assumed that perft value will fit in remaining (~hashMask) bits
            hashTable[hash & indexBits] = (hash  & hashBits) | (perftNminus1) ;
#if PRINT_HASH_STATS == 1
            atomicAdd(&numStores[depth], 1);
#endif
        }
    }
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

    if (index < nThreads && pos.whitePieces)    // pos.whitePieces == 0 indicates an invalid board (hash hit)
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
        makemove_and_count_moves_single_level<true><<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, firstLevelChildBoards, (uint32 *) secondLevelMoveCounts, numFirstLevelMoves);

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
        // Ankan - this won't work with 64 bit build!
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
        makemove_and_count_moves_single_level<true><<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, shMem.allFirstLevelChildBoards, secondLevelMoveCounts, numFirstLevelMoves);
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
        // allocate memory for:
#if COMBINE_DEVICE_MALLOCS == 1
        void **pointers[5];
        int sizes[5];
        pointers[0] = (void **) &shMem.allFirstLevelChildMoves;     // first level child moves
        pointers[1] = (void **) &shMem.currentLevelBoards;          // current level board positions
        pointers[2] = (void **) &shMem.boardPointers;               // and also for pointers for first level childs to to current level board positions
        pointers[3] = (void **) &shMem.allFirstLevelChildBoards;    // first level child boards
        pointers[4] = (void **) &shMem.allSecondLevelMoveCounts;    // memory to hold counts of moves that will be generated by first level boards

        sizes[0] = sizeof(CMove) * numFirstLevelMoves;
        sizes[1] = sizeof(HexaBitBoardPosition) * BLOCK_SIZE;
        sizes[2] = sizeof(HexaBitBoardPosition *) * numFirstLevelMoves;
        sizes[3] = sizeof(HexaBitBoardPosition) * numFirstLevelMoves;
        int size = sizeof(uint32) * numFirstLevelMoves;
        // (add some scratch space needed by scan and intervalExpand functions)
        size = (int) size * 1.2f + 256;
        sizes[4] = size;

        deviceMultiAlloc(pointers, sizes, 5);
#else
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
#endif
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
        makemove_and_count_moves_single_level<true><<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelBoardPointers, firstLevelChildMoves, firstLevelChildBoards, (uint32 *) secondLevelMoveCounts, numFirstLevelMoves);

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

        // Ankan - this will also not work with 64 bit
        // TODO: move to base + array of indices method, instead of array of pointers
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
        makemove_and_count_moves_single_level<true><<<nBlocks, BLOCK_SIZE, 0, childStream>>>(secondLevelBoardPointers, secondLevelChildMoves, secondLevelChildBoards, (uint32 *) thirdLevelMoveCounts, numSecondLevelMoves);

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
        // Ankan - won't work on 64 bit! Switch to base + indices
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
        pos = *(positions[index]);
        color = pos.chance;


        if (moves != NULL)
        {
            move = moves[index];
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
// if move is NULL, the function is supposed to return perft of the current position (pos)
// otherwise, it will first make the move and then return perft of the resulting position
__device__ void perft_bb_gpu_recursive_launcher(HexaBitBoardPosition **posPtr, CMove *move, uint64 *globalPerftCounter, 
                                                int depth, CMove *movesStack, HexaBitBoardPosition *boardStack,
                                                HexaBitBoardPosition **boardPtrStack, int launchDepth)
{
    HexaBitBoardPosition *pos = *posPtr;
    uint32 nMoves = 0;
    uint8 color = pos->chance;
    if (depth == 1)
    {
        if (move != NULL)
        {
            makeMove(pos, *move, color);
            color = !color;
        }
        nMoves = countMoves(pos, color);
        atomicAdd (globalPerftCounter, nMoves);
    }
    else if (depth <= launchDepth)
    {
        perft_bb_gpu_safe<<<1, BLOCK_SIZE, sizeof(sharedMemAllocs), 0>>> (posPtr, move, globalPerftCounter, depth, 1);
        cudaDeviceSynchronize();

        // 'free' up the memory used by the launch
        //    printf("\nmemory used by previous parallel launch: %d bytes\n", preAllocatedMemoryUsed);
        preAllocatedMemoryUsed = 0;
    }
    else
    {
        // recurse serially till we reach a depth where we can launch parallel work
        //nMoves = generateBoards(pos, color, boardStack);
        if (move != NULL)
        {
            makeMove(pos, *move, color);
            color = !color;
        }
        nMoves = generateMoves(pos, color, movesStack);
        *boardPtrStack = boardStack;
        for (uint32 i=0; i < nMoves; i++)
        {
            *boardStack = *pos;
            perft_bb_gpu_recursive_launcher(boardPtrStack, &movesStack[i], globalPerftCounter, depth - 1, 
                                            &movesStack[MAX_MOVES],  boardStack + 1, boardPtrStack + 1, launchDepth);
        }
    }
}

// the starting kernel for perft
__global__ void perft_bb_driver_gpu(HexaBitBoardPosition *pos, uint64 *globalPerftCounter, int depth, void *serialStack, void *devMemory, int launchDepth)
{
    // set device memory pointer
    preAllocatedBuffer = devMemory;
    preAllocatedMemoryUsed = 0;

    // call the recursive function
    // Three items are stored in the stack
    // 1. the board position pointer (one item per level)
    // 2. the board position         (one item per level)
    // 3. generated moves            (upto MAX_MOVES item per level)
    HexaBitBoardPosition *boardStack        = (HexaBitBoardPosition *)  serialStack;
    HexaBitBoardPosition **boardPtrStack    = (HexaBitBoardPosition **) ((char *)serialStack + (16 * 1024));
    CMove *movesStack                       = (CMove *)                 ((char *)serialStack + (20 * 1024));

#if COUNT_NUM_COUNT_MOVES == 1
    numCountMoves = 0ull;
#endif

    *boardPtrStack = pos;   // put the given board in the board ptr stack
    perft_bb_gpu_recursive_launcher(boardPtrStack, NULL, globalPerftCounter, depth, movesStack, boardStack, boardPtrStack + 1, launchDepth);

#if COUNT_NUM_COUNT_MOVES == 1
    printf("Total no. of times countMoves was called: %llu \n", numCountMoves);
#endif
}



//--------------------------------------
// transposition table helper functions
//--------------------------------------

#if USE_TRANSPOSITION_TABLE == 1

// look up the transposition table for an entry
__device__ __forceinline__ TT_Entry lookupTT(uint64 hash, TT_Entry *TT, uint64 indexBits)
{
#ifdef __CUDA_ARCH__
        // stupid CUDA compiler can't see that I need 128 bit atomic reads/writes!
#if USE_DUAL_SLOT_TT == 1
        uint4 mostRecent = TT[hash & (indexBits)].mostRecent.rawVal;
        uint4 deepest    = TT[hash & (indexBits)].deepest.rawVal;
        TT_Entry entry;
        entry.mostRecent.rawVal = mostRecent;
        entry.deepest.rawVal = deepest;
#else
        uint4 val = TT[hash & (indexBits)].rawVal;
        TT_Entry entry;
        entry.rawVal = val;
#endif
        return entry;
#else
        return TT[hash & (indexBits)];
#endif
}

// check if the given position is present in transposition table entry
__device__ __forceinline__ bool searchTTEntry(TT_Entry &entry, uint64 hash, uint64 *perft, uint64 hashBits)
{
#if USE_DUAL_SLOT_TT == 1
    if ((entry.mostRecent.hashKey & hashBits) == (hash & hashBits))
    {
        *perft = entry.mostRecent.perftVal;
        return true;
    }
    if ((entry.deepest.hashKey & hashBits) == (hash & hashBits))
    {
        *perft = entry.deepest.perftVal;
        return true;
    }
#else
    if ((entry.hashKey & hashBits) == (hash & hashBits))
    {
        *perft = entry.perftVal;
        return true;
    }
#endif

    return false;
}

__device__ __forceinline__ void storeUpdatedTTEntry(TT_Entry &entry, uint64 hash, TT_Entry *TT, uint64 indexBits)
{
#ifdef __CUDA_ARCH__
#if USE_DUAL_SLOT_TT == 1
        TT[hash & (indexBits)].mostRecent.rawVal = entry.mostRecent.rawVal;
        TT[hash & (indexBits)].deepest.rawVal    = entry.deepest.rawVal;
#else
        TT[hash & (indexBits)].rawVal = entry.rawVal;
#endif
#else
        TT[hash & (indexBits)] = entry;
#endif
    
}

__device__ __forceinline__ void storeTTEntry(TT_Entry &entry, uint64 hash, int depth, uint64 count, TT_Entry *TT, uint64 indexBits)
{
#if USE_DUAL_SLOT_TT == 1

    // add this pos to deepest slot if this is deeper than deepest, or if the deepest is empty (depth=0)
    if (entry.deepest.depth <= depth)
    {
        // avoid the entry to get overwritten if most recent slot is free (or is at lower depth)
        if (entry.mostRecent.depth < entry.deepest.depth)
        {
            entry.mostRecent = entry.deepest;
        }

        entry.deepest.perftVal = count;
        entry.deepest.hashKey = hash;
        entry.deepest.depth = depth;
        storeUpdatedTTEntry(entry, hash, TT, indexBits);
    }
    else
    {
        // otherwise add it to mostRecent slot
        entry.mostRecent.perftVal = count;
        entry.mostRecent.hashKey = hash;
        entry.mostRecent.depth = depth;
        storeUpdatedTTEntry(entry, hash, TT, indexBits);
    }
#else
    // only replace hash table entry if previously stored entry is at shallower depth
    if (entry.depth <= depth)
    {
        entry.perftVal = count;
        entry.hashKey = hash;
        entry.depth = depth;
        storeUpdatedTTEntry(entry, hash, TT, indexBits);
    }
#endif
}
#endif



//--------------------------------------------------------------------------------------------------
// versions of the above kernel that use hash tables
//--------------------------------------------------------------------------------------------------
#if USE_TRANSPOSITION_TABLE == 1

// positions is array of pointers containing the old position on which moves[] is to be made
// hashes[] array contains hash (duplicated) of positions array[] before making the move
// perftCounts[] is array of pointers to perft counters
__global__ void perft_bb_gpu_depth3_hash(HexaBitBoardPosition **positions, uint64 *hashes, CMove *moves, uint64 **perftCounters, int nThreads, TTInfo* TTs)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos;
    uint64 hash;
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
        hash = makeMoveAndUpdateHash(&pos, hashes[index], move, color);

        color = !color;
        uint64 entry = TTs->depth3[hash & TTs->indexBits[3]];
        if ((entry & TTs->hashBits[3]) == (hash & TTs->hashBits[3]))
        {
            // hash hit
            atomicAdd(perftCounters[index], entry & TTs->indexBits[3]);
        }
        else
        {
            nMoves = countMoves(&pos, color);
        }
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // number of first level moves
    uint32 numFirstLevelMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (numFirstLevelMoves == 0)
        return;

    // first thread of block allocates memory to store all moves generated by the thread block
    if (threadIdx.x == 0)
    {
        // TODO: maybe combine these multiple memory requests to a single request (to save on no. of AtmoicAdds() used)
        // allocate memory for:
#if COMBINE_DEVICE_MALLOCS == 1
        void **pointers[9];
        int sizes[9];
        pointers[0] = (void **) &shMem.allFirstLevelChildMoves;     // first level child moves
        pointers[1] = (void **) &shMem.currentLevelBoards;          // current level board positions
        pointers[2] = (void **) &shMem.currentLevelHashes;          // current level hashes
        pointers[3] = (void **) &shMem.perft3Counters;              // perft counters for depth 3 (equal to no. of threads in this thread block)
        pointers[4] = (void **) &shMem.firstToCurrentLevelIndices;  // and also for indices for first level childs to to current level board positions
        pointers[5] = (void **) &shMem.perft2Counters;              // perft counters for depth 2
        pointers[6] = (void **) &shMem.allFirstLevelChildBoards;    // first level child boards
        pointers[7] = (void **) &shMem.firstLevelHashes;            // and their hashes
        pointers[8] = (void **) &shMem.allSecondLevelMoveCounts;    // memory to hold counts of moves that will be generated by first level boards

        sizes[0] = sizeof(CMove) * numFirstLevelMoves;
        sizes[1] = sizeof(HexaBitBoardPosition) * BLOCK_SIZE;
        sizes[2] = sizeof(uint64) * BLOCK_SIZE;
        sizes[3] = sizeof(uint32) * BLOCK_SIZE;
        sizes[4] = sizeof(int) * numFirstLevelMoves;
        sizes[5] = sizeof(uint32) * numFirstLevelMoves;
        sizes[6] = sizeof(HexaBitBoardPosition) * numFirstLevelMoves;
        sizes[7] = sizeof(uint64) * numFirstLevelMoves;

        int size = sizeof(uint32) * numFirstLevelMoves;
        size = (int) size * 1.2f + 256;
        // (add some scratch space needed by scan and intervalExpand functions)
        sizes[8] = size;
        deviceMultiAlloc(pointers, sizes, 9);
#else
        // first level child moves
        deviceMalloc(&shMem.allFirstLevelChildMoves, sizeof(CMove) * numFirstLevelMoves);

        // current level board positions
        deviceMalloc(&shMem.currentLevelBoards, sizeof(HexaBitBoardPosition) * BLOCK_SIZE);

        // current level hashes
        deviceMalloc(&shMem.currentLevelHashes, sizeof(uint64) * BLOCK_SIZE);

        // perft counters for depth 3 (equal to no. of threads in this thread block)
        deviceMalloc(&shMem.perft3Counters, sizeof(uint32) * BLOCK_SIZE);

        // and also for indices for first level childs to to current level board positions
        deviceMalloc(&shMem.firstToCurrentLevelIndices, sizeof(int) * numFirstLevelMoves);

        // perft counters for depth 2
        deviceMalloc(&shMem.perft2Counters, sizeof(uint32) * numFirstLevelMoves);

        // also need to allocate first level child boards
        deviceMalloc(&shMem.allFirstLevelChildBoards, sizeof(HexaBitBoardPosition) * numFirstLevelMoves);

        // and hashes for them
        deviceMalloc(&shMem.firstLevelHashes, sizeof(uint64) * numFirstLevelMoves);

        // allocate memory to hold counts of moves that will be generated by first level boards
        // first level move counts is in per-thread variable "nMoves"
        int size = sizeof(uint32) * numFirstLevelMoves;
        size = (int) size * 1.2f + 256;
        // (add some scratch space needed by scan and intervalExpand functions)
        deviceMalloc(&shMem.allSecondLevelMoveCounts, size);
#endif
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *currentLevelBoards = shMem.currentLevelBoards;
    uint64               *currentLevelHashes = shMem.currentLevelHashes;
    currentLevelBoards[threadIdx.x] = pos;
    currentLevelHashes[threadIdx.x] = hash;

    // make all the first to current level indices point to the current board
    int *firstToCurrentLevelIndices = shMem.firstToCurrentLevelIndices + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        firstToCurrentLevelIndices[i] = threadIdx.x;
    }

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, firstLevelChildMoves);
    }

    // clear the perft counters
    shMem.perft3Counters[threadIdx.x] = 0;

    __syncthreads();


    int *secondLevelMoveCounts = (int *) shMem.allSecondLevelMoveCounts;
    if (threadIdx.x == 0)
    {
        HexaBitBoardPosition *firstLevelChildBoards;
        cudaStream_t childStream;
        uint32 nBlocks;

        firstLevelChildBoards = shMem.allFirstLevelChildBoards;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

        // 1. count the moves that would be generated by childs
        //    (and also generate first level child boards, from child boards)
        nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;

        makemove_and_count_moves_single_level_hash<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
            (currentLevelBoards, currentLevelHashes, shMem.perft3Counters, 
             shMem.firstToCurrentLevelIndices, firstLevelChildMoves, 
             TTs->depth2, TTs->hashBits[2], TTs->indexBits[2], 
             firstLevelChildBoards, shMem.firstLevelHashes, (uint32 *) secondLevelMoveCounts, 
             shMem.perft2Counters, numFirstLevelMoves, 2);

        // 2. secondLevelMoveCounts now has individual move counts, run a scan on it
        int *pNumSecondLevelMoves = secondLevelMoveCounts + numFirstLevelMoves;
        int *scratchSpace = pNumSecondLevelMoves + 1;
        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (secondLevelMoveCounts, numFirstLevelMoves, secondLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	         pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();
        // the scan also gives the total of moveCounts
        int numSecondLevelMoves = *pNumSecondLevelMoves;

        //printf("\nAfter calling scanD, numSecondLevelMoves = %d\n", numSecondLevelMoves);

        if (numSecondLevelMoves)
        {
            // shMem.movesForThread[] has scan of how many second level moves each thread of this thread block will generate
            // we use that as input for the intervalExpand kernel to generate secondToFirstLevelIndices

            // 3. allocate memory for:
            // second level child moves
            CMove *secondLevelChildMoves;
            deviceMalloc(&secondLevelChildMoves, sizeof(CMove) * numSecondLevelMoves);

            // and board indices to first level boards
            int *secondToFirstLevelIndices;
            deviceMalloc(&secondToFirstLevelIndices, sizeof(int) * numSecondLevelMoves);

            // Generate secondToFirstLevelIndices by running interval expand
            // 
            // Expand numFirstLevelMoves items -> secondLevelMoveCounts items
            // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output

            // secondLevelMoveCounts now have the exclusive scan - containing the indices to put moves on

            mgpu::IntervalExpandDGenValues(numSecondLevelMoves, secondLevelMoveCounts, (int) 0,
                                           1, numFirstLevelMoves, secondToFirstLevelIndices, 
                                           childStream, scratchSpace);

            // 5. generate the second level child moves
            // secondLevelMoveCounts is used by the below kernel to index into secondLevelChildMoves[] - to know where to put the generated moves
            generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelChildBoards, secondLevelChildMoves, secondLevelMoveCounts, numFirstLevelMoves);


            // 6. now we have all second level generated moves in secondLevelChildMoves .. launch a kernel at depth - 2 to make the moves and count leaves
            nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;
            makeMove_and_perft_single_level_indices<<<nBlocks, BLOCK_SIZE, 0, childStream>>> 
                (firstLevelChildBoards, shMem.perft2Counters, secondToFirstLevelIndices, secondLevelChildMoves, numSecondLevelMoves);

            // 7. launch a kernel to compute perft3 values from perft2 values.. and to update perft2 values in hash
            nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;
            calcPerftNFromPerftNminus1<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                (shMem.perft3Counters, firstToCurrentLevelIndices, shMem.perft2Counters, shMem.firstLevelHashes, 
                 TTs->depth2, TTs->hashBits[2], TTs->indexBits[2], numFirstLevelMoves, 2);

            cudaDeviceSynchronize();
        }

        cudaStreamDestroy(childStream);
    }

    __syncthreads();
    if (nMoves)
    {
        uint64 perftVal = shMem.perft3Counters[threadIdx.x];
        atomicAdd(perftCounters[index], perftVal);

        // store in hash table
        TTs->depth3[hash & TTs->indexBits[3]] = (hash       & TTs->hashBits[3])  |
                                                (perftVal   & TTs->indexBits[3]) ;
    }
}

// same as above kernel but launches the main kernel to process 3 levels/plys in single launch
// positions is array of pointers containing the old position on which moves[] is to be made
// hashes[] array contains hash (duplicated) of positions array[] before making the move
// perftCounts[] is array of pointers to perft counters
__global__ void perft_bb_gpu_depth4_hash(HexaBitBoardPosition **positions, uint64 *hashes, CMove *moves, uint64 **perftCounters, int nThreads, TTInfo *TTs)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos;
    CMove move;
    uint8 color;
    uint64 hash;

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        move = moves[index];
        color = pos.chance;
        hash = makeMoveAndUpdateHash(&pos, hashes[index], move, color);

        color = !color;

        // look up the transposition table
#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[4], 1);
#endif

        uint64 entry = TTs->depth4[hash & TTs->indexBits[4]];        
        if ((entry & TTs->hashBits[4]) == (hash & TTs->hashBits[4]))
        {
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[4], 1);
#endif
            atomicAdd(perftCounters[index], entry & TTs->indexBits[4]);
        }
        else
        {
            nMoves = countMoves(&pos, color);
        }
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // number of first level moves
    uint32 numFirstLevelMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (numFirstLevelMoves == 0)
        return;

    // some terminology to avoid confusion:
    //  - we are at depth = 4, so currentLevelBoards, currentLevelHashes, etc are for positions of depth 4
    //  - firstLevel is depth = 3, so for e.g, numFirstLevelMoves is sum of num of moves 
    //                                         generated by all child boards of current boards
    //  - secondLevel is depth = 2
    //  - thirdLevel is depth = 1 (i.e, the leaves where we just count the moves by calling the final kernel)

    // first thread of block allocates memory to store all moves generated by the thread block
    if (threadIdx.x == 0)
    {
        // allocate memory for:
#if COMBINE_DEVICE_MALLOCS == 1
        void **pointers[9];
        int sizes[9];
        pointers[0] = (void **) &shMem.allFirstLevelChildMoves;     // first level child moves
        pointers[1] = (void **) &shMem.currentLevelBoards;          // current level board positions
        pointers[2] = (void **) &shMem.currentLevelHashes;          // current level hashes
        pointers[3] = (void **) &shMem.perft4Counters;              // perft counters for depth 4 (equal to no. of threads in this thread block)
        pointers[4] = (void **) &shMem.firstToCurrentLevelIndices;  // and also for indices for first level childs to to current level board positions
        pointers[5] = (void **) &shMem.perft3Counters;              // perft counters for depth 3
        pointers[6] = (void **) &shMem.allFirstLevelChildBoards;    // first level child boards
        pointers[7] = (void **) &shMem.firstLevelHashes;            // and their hashes
        pointers[8] = (void **) &shMem.allSecondLevelMoveCounts;    // memory to hold counts of moves that will be generated by first level boards

        sizes[0] = sizeof(CMove) * numFirstLevelMoves;
        sizes[1] = sizeof(HexaBitBoardPosition) * BLOCK_SIZE;
        sizes[2] = sizeof(uint64) * BLOCK_SIZE;
        sizes[3] = sizeof(uint32) * BLOCK_SIZE;
        sizes[4] = sizeof(int) * numFirstLevelMoves;
        sizes[5] = sizeof(uint32) * numFirstLevelMoves;
        sizes[6] = sizeof(HexaBitBoardPosition) * numFirstLevelMoves;
        sizes[7] = sizeof(uint64) * numFirstLevelMoves;

        int size = sizeof(uint32) * numFirstLevelMoves;
        size = (int) size * 1.2f + 256;
        // (add some scratch space needed by scan and intervalExpand functions)
        sizes[8] = size;
        deviceMultiAlloc(pointers, sizes, 9);
#else
        // first level child moves
        deviceMalloc(&shMem.allFirstLevelChildMoves, sizeof(CMove) * numFirstLevelMoves);

        // current level board positions
        deviceMalloc(&shMem.currentLevelBoards, sizeof(HexaBitBoardPosition) * BLOCK_SIZE);

        // current level hashes
        deviceMalloc(&shMem.currentLevelHashes, sizeof(uint64) * BLOCK_SIZE);

        // perft counters for depth 4 (equal to no. of threads in this thread block)
        deviceMalloc(&shMem.perft4Counters, sizeof(uint32) * BLOCK_SIZE);

        // and also for indices for first level childs to to current level board positions
        deviceMalloc(&shMem.firstToCurrentLevelIndices, sizeof(int) * numFirstLevelMoves);

        // perft counters for depth 3 (for the first level childs)
        deviceMalloc(&shMem.perft3Counters, sizeof(uint32) * numFirstLevelMoves);

        // also need to allocate first level child boards
        deviceMalloc(&shMem.allFirstLevelChildBoards, sizeof(HexaBitBoardPosition) * numFirstLevelMoves);

        // and hashes for them
        deviceMalloc(&shMem.firstLevelHashes, sizeof(uint64) * numFirstLevelMoves);

        // allocate memory to hold counts of moves that will be generated by first level boards
        // first level move counts is in per-thread variable "nMoves"
        int size = sizeof(uint32) * numFirstLevelMoves;
        size = (int) size * 1.2f + 256;
        // (add some scratch space needed by scan and intervalExpand functions)
        deviceMalloc(&shMem.allSecondLevelMoveCounts, size);
#endif
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *currentLevelBoards = shMem.currentLevelBoards;
    uint64               *currentLevelHashes = shMem.currentLevelHashes;
    currentLevelBoards[threadIdx.x] = pos;
    currentLevelHashes[threadIdx.x] = hash;

    // make all the first to current level indices point to the current board
    int *firstToCurrentLevelIndices = shMem.firstToCurrentLevelIndices + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        firstToCurrentLevelIndices[i] = threadIdx.x;
    }

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, firstLevelChildMoves);
    }

    // clear the perft counters for the current level
    shMem.perft4Counters[threadIdx.x] = 0;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        //printf("\nIn the depth 4 hash kernel!\n");

        HexaBitBoardPosition *firstLevelChildBoards;
        int *secondLevelMoveCounts = (int *) shMem.allSecondLevelMoveCounts;
        cudaStream_t childStream;
        uint32 nBlocks;

        firstLevelChildBoards = shMem.allFirstLevelChildBoards;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

        // 1. count the moves that would be generated by childs
        //    (and also generate first level child boards, from child boards)
        nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;

        makemove_and_count_moves_single_level_hash<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
            (currentLevelBoards, currentLevelHashes, shMem.perft4Counters, 
             shMem.firstToCurrentLevelIndices, firstLevelChildMoves, 
             TTs->depth3, TTs->hashBits[3], TTs->indexBits[3], 
             firstLevelChildBoards, shMem.firstLevelHashes, (uint32 *) secondLevelMoveCounts, 
             shMem.perft3Counters, numFirstLevelMoves, 3);

        //cudaDeviceSynchronize();
        //printf("\nAfter calling makemove_and_count_moves_single_level_hash!\n");

        // 2. secondLevelMoveCounts now has individual move counts, run a scan on it
        int *pNumSecondLevelMoves = secondLevelMoveCounts + numFirstLevelMoves;
        int *scratchSpace = pNumSecondLevelMoves + 1;
        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (secondLevelMoveCounts, numFirstLevelMoves, secondLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	         pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();
        // the scan also gives the total of moveCounts
        int numSecondLevelMoves = *pNumSecondLevelMoves;

        if (numSecondLevelMoves)
        {
            // 3. allocate memory for:
            // second level child moves
            CMove *secondLevelChildMoves;
            deviceMalloc(&secondLevelChildMoves, sizeof(CMove) * numSecondLevelMoves);

            // and board indices to first level boards
            int *secondToFirstLevelIndices;
            deviceMalloc(&secondToFirstLevelIndices, sizeof(int) * numSecondLevelMoves);

            // Generate secondToFirstLevelIndices by running interval expand
            // 
            // Expand numFirstLevelMoves items -> secondLevelMoveCounts items
            // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output

            // secondLevelMoveCounts now have the exclusive scan - containing the indices to put moves on

            mgpu::IntervalExpandDGenValues(numSecondLevelMoves, secondLevelMoveCounts, (int) 0,
                                           1, numFirstLevelMoves, secondToFirstLevelIndices, 
                                           childStream, scratchSpace);

            // 5. generate the second level child moves
            // secondLevelMoveCounts is used by the below kernel to index into secondLevelChildMoves[] - to know where to put the generated moves
            generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(firstLevelChildBoards, secondLevelChildMoves, secondLevelMoveCounts, numFirstLevelMoves);


            //cudaDeviceSynchronize();
            //printf("\nAfter calling generate_moves_single_level to generate second level moves!\n");


            // 6. now we have all second level generated moves in secondLevelChildMoves. We now want to
            //  count the number of moves that would be generated by second level childs
            //  (and also generate second level child boards, from first level child board pointers and second level secondLevelChildMoves)

            // firstly allocate space to hold them
            HexaBitBoardPosition *secondLevelChildBoards;
            uint64 *secondLevelHashes;
            int *thirdLevelMoveCounts;

            deviceMalloc(&secondLevelChildBoards, sizeof(HexaBitBoardPosition) * numSecondLevelMoves);
            deviceMalloc(&secondLevelHashes, sizeof(uint64) * numSecondLevelMoves);
            // perft counters for depth 2 (second level boards)
            deviceMalloc(&shMem.perft2Counters, sizeof(uint32) * numSecondLevelMoves);

            int sizeAlloc = (int) sizeof(uint32) * numSecondLevelMoves * 1.2f + 256;
            deviceMalloc(&thirdLevelMoveCounts, sizeAlloc);
            nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;

            
            makemove_and_count_moves_single_level_hash<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                (firstLevelChildBoards, shMem.firstLevelHashes, shMem.perft3Counters, 
                 secondToFirstLevelIndices, secondLevelChildMoves, 
                 TTs->depth2, TTs->hashBits[2], TTs->indexBits[2], 
                 secondLevelChildBoards, secondLevelHashes, (uint32 *) thirdLevelMoveCounts, 
                 shMem.perft2Counters, numSecondLevelMoves, 2);

            //cudaDeviceSynchronize();
            //printf("\nAfter calling makemove_and_count_moves_single_level_hash to generate second level boards, and do many things!\n");


            // 7. thirdLevelMoveCounts now has individual move counts, run a scan on it
            int *pNumThirdLevelMoves = thirdLevelMoveCounts + numSecondLevelMoves;
            int *scratchSpace2 = pNumThirdLevelMoves + 1;
            mgpu::ScanD<mgpu::MgpuScanTypeExc>
                (thirdLevelMoveCounts, numSecondLevelMoves, thirdLevelMoveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(), 
	             pNumThirdLevelMoves, false, childStream, scratchSpace2);

            cudaDeviceSynchronize();
            // the scan also gives the total of moveCounts
            int numThirdLevelMoves = *pNumThirdLevelMoves;

            //cudaDeviceSynchronize();
            //printf("\nAfter Running scan the second time to figure out numThirdLevelMoves: %d\n", numThirdLevelMoves);


            // 8. allocate memory for:
            // third level child moves
            CMove *thirdLevelChildMoves;
            deviceMalloc(&thirdLevelChildMoves, sizeof(CMove) * numThirdLevelMoves);
            // and indices from third to second level boards
            int *thirdToSecondLevelIndices;
            deviceMalloc(&thirdToSecondLevelIndices, sizeof(int) * numThirdLevelMoves);
            
            // 9. now run interval expand to get thirdToSecondLevelIndices filled
            mgpu::IntervalExpandDGenValues(numThirdLevelMoves, thirdLevelMoveCounts, (int) 0,
                                           1, numSecondLevelMoves, thirdToSecondLevelIndices, 
                                           childStream, scratchSpace2);


            //cudaDeviceSynchronize();
            //printf("\nAfter Running IntervalExpandDGenValues the second time\n");


            // 10. Generate the third level child moves
            // thirdLevelMoveCounts (containing the scan) is used by the below kernel 
            // to index into thirdLevelChildMoves[] - to know where to put the generated moves
            generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                (secondLevelChildBoards, thirdLevelChildMoves, thirdLevelMoveCounts, numSecondLevelMoves);
            
            //cudaDeviceSynchronize();
            //printf("\njust before running the final kernel...\n");

            // 11. finally run countMoves on third level moves!
            // this will populate perft2Counters
            nBlocks = (numThirdLevelMoves - 1) / BLOCK_SIZE + 1;
            makeMove_and_perft_single_level_indices<<<nBlocks, BLOCK_SIZE, 0, childStream>>> 
                (secondLevelChildBoards, shMem.perft2Counters, thirdToSecondLevelIndices, thirdLevelChildMoves, numThirdLevelMoves);

            //cudaDeviceSynchronize();
            //printf("\nfinished the big kernel...updating hash and counters now\n");


            // 12. launch a kernel to compute perft3 values from perft2 values.. 
            //     and to update perft2 values in hash
            nBlocks = (numSecondLevelMoves - 1) / BLOCK_SIZE + 1;
            calcPerftNFromPerftNminus1<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                (shMem.perft3Counters, secondToFirstLevelIndices, shMem.perft2Counters, secondLevelHashes, 
                 TTs->depth2, TTs->hashBits[2], TTs->indexBits[2], numSecondLevelMoves, 2);


            //cudaDeviceSynchronize();
            //printf("\nperft2 to perft3 done.. running perft3 to perft4\n");


            // 13. launch a kernel to compute perft4 values from perft3 values
            nBlocks = (numFirstLevelMoves - 1) / BLOCK_SIZE + 1;
            calcPerftNFromPerftNminus1<<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                (shMem.perft4Counters, firstToCurrentLevelIndices, shMem.perft3Counters, shMem.firstLevelHashes,
                 TTs->depth3, TTs->hashBits[3], TTs->indexBits[3], numFirstLevelMoves, 3);


            //cudaDeviceSynchronize();
            //printf("\nEverything done... whats the problem then?\n");

            cudaDeviceSynchronize();
        }

        cudaStreamDestroy(childStream);
    }

    __syncthreads();
    if (nMoves)
    {
        uint64 perftVal = shMem.perft4Counters[threadIdx.x];

        // add it to perftCounter for this thread... and store it in TT
        atomicAdd(perftCounters[index], perftVal);

        // store perft val in transposition table
#if PRINT_HASH_STATS == 1
        atomicAdd(&numStores[4], 1);
#endif

        TTs->depth4[hash & TTs->indexBits[4]] =  (hash       & TTs->hashBits [4]) |
                                                 (perftVal   & TTs->indexBits[4]) ;
    }
}



// positions is array of pointers containing the old position on which moves[] is to be made
// hashes[] is the old hash before making the move
// perftCounts[] is array of pointers to perft counters

// this function is responsible for checking if the position after making the move exists in the hash table
// and also to update the positions hash value in the hash table (if needed)
__global__ void perft_bb_gpu_main_hash(HexaBitBoardPosition **positions,  uint64 *hashes, CMove *moves, 
                                       uint64 **perftCounters, int depth, int nThreads, TTInfo* TTs)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos;
    uint64 *perftCounter;
    CMove move;
    uint8 color;
    uint64 hash, lookupHash;
    TT_Entry entry;


    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    // perft for the board this thread is processing at this depth
    uint64 perftVal = 0;

    if (index < nThreads)
    {
        pos = *(positions[index]);
        perftCounter = perftCounters[index];
        color = pos.chance;
        hash = hashes[index];


        if (moves != NULL)
        {
            move = moves[index];
            hash = makeMoveAndUpdateHash(&pos, hash, move, color);
            color = !color;
        }

        // look up the transposition table

#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[depth], 1);
#endif

        lookupHash = hash ^ (ZOB_KEY(depth) * depth);
        entry = lookupTT(lookupHash, TTs->deepTT[depth], TTs->indexBits[depth]);
        if (searchTTEntry(entry, lookupHash, &perftVal, TTs->hashBits[depth]))
        {
            // hash hit: no need to process this position further
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[depth], 1);
#endif
            // TODO: might want to do a warp wide reduction before atomic add
            atomicAdd(perftCounter, perftVal);
        }
        else
        {
            nMoves = countMoves(&pos, color);
        }
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

        // perft counters (equal to no. of threads in this thread block)
        deviceMalloc(&shMem.perftCounters, sizeof(uint64) * BLOCK_SIZE);
        //printf("\nPerftCounters: %X\n", shMem.perftCounters);

        // and pointers to perft counters (equal to no  of child boards generated by threads in the block)
        deviceMalloc(&shMem.counterPointers, sizeof(uint64 *) * numFirstLevelMoves);

        // and hashes (with lot of duplicacy)
        deviceMalloc(&shMem.firstLevelHashes, sizeof(uint64) * numFirstLevelMoves);
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    CMove *firstLevelChildMoves = shMem.allFirstLevelChildMoves + moveListOffset;

    // update the current level board with the board after makeMove
    HexaBitBoardPosition *boards = shMem.currentLevelBoards;
    boards[threadIdx.x] = pos;

    // initialize the perft counters
    uint64 *counters = shMem.perftCounters;
    counters[threadIdx.x] = 0;

    // make all the board pointers point to the current board
    // ... and counter pointers point to correct counter
    HexaBitBoardPosition **boardPointers = shMem.boardPointers + moveListOffset;
    uint64 **counterPointers = shMem.counterPointers + moveListOffset;
    uint64 *nextHashes = shMem.firstLevelHashes + moveListOffset;
    for (int i=0 ; i < nMoves; i++)
    {
        boardPointers[i] = &boards[threadIdx.x];
        counterPointers[i] = &counters[threadIdx.x];
        nextHashes[i] = hash;
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
        {
            perft_bb_gpu_single_level<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, firstLevelChildMoves, perftCounter, numFirstLevelMoves);
        }

#if PARALLEL_LAUNCH_LAST_3_LEVELS == 1
        else if (depth == 5)
            perft_bb_gpu_depth4_hash<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, nextHashes, firstLevelChildMoves, counterPointers, numFirstLevelMoves, TTs);
#endif
        else if (depth == 4)
            perft_bb_gpu_depth3_hash<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, nextHashes, firstLevelChildMoves, counterPointers, numFirstLevelMoves, TTs);
        else
        {
            perft_bb_gpu_main_hash<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (boardPointers, nextHashes, firstLevelChildMoves, counterPointers, depth-1, numFirstLevelMoves, TTs);
        }

        cudaDeviceSynchronize();
        cudaStreamDestroy(childStream);
    }

    __syncthreads();

    if (nMoves && depth > 2)    // for depth 2, everything is already done
    {
        // now all the child perfts have been populated in (counters[threadIdx.x])
        perftVal = counters[threadIdx.x];

        // add it to perftCounter for this thread... and store it in TT
        atomicAdd(perftCounter, perftVal);

#if PRINT_HASH_STATS == 1
        atomicAdd(&numStores[depth], 1);
#endif

        // store perft val in transposition table
        storeTTEntry(entry, lookupHash, depth, perftVal, TTs->deepTT[depth], TTs->indexBits[depth]);
    }
}



// traverse the tree recursively (and serially) and launch parallel work on reaching launchDepth
__device__ uint64 perft_bb_gpu_hash_recursive_launcher(HexaBitBoardPosition **posPtr, uint64 hash, CMove *move, 
                                                       uint64 *globalPerftCounter, int depth, CMove *movesStack, HexaBitBoardPosition *boardStack,
                                                       HexaBitBoardPosition **boardPtrStack, int launchDepth, TTInfo *TTs)
{
    HexaBitBoardPosition *pos   = *posPtr;
    uint8 color                 = pos->chance;
    uint32 nMoves               = 0;
    uint64 perftVal             = 0;

    if (depth == 1)
    {
        if (move != NULL)
        {
            makeMove(pos, *move, color);
            color = !color;
        }
        perftVal = countMoves(pos, color);
    }
    else if (depth <= launchDepth)
    {
        // TODO: it's easily possible to save one launch depth by launching this kernel with nMoves (i.e, below) instead of 1 move
        *globalPerftCounter = 0;
        
        // put the hash in scratch storage and pass it's pointer (use boardStack as scratch space)
        uint64 *pHash = (uint64 *) boardStack;
        *pHash = hash;

        // also put the pointer to globalPerftCounter and pass it on
        uint64 **pPerftCounter = (uint64 **) (pHash + 1);
        *pPerftCounter = globalPerftCounter;
        
        // put the Transposition table info structure also in memory to pass it on
        TTInfo *pTTs = (TTInfo *) (pHash + 16);	// beware of alignment issues !!
        *pTTs = *TTs;

        perft_bb_gpu_main_hash <<<1, BLOCK_SIZE, sizeof(sharedMemAllocs), 0>>> 
                               (posPtr, pHash, move, pPerftCounter, depth, 1, pTTs);
        cudaDeviceSynchronize();

        // 'free' up the memory used by the launch
        // printf("\nmemory used by previous parallel launch: %d bytes\n", preAllocatedMemoryUsed);
        preAllocatedMemoryUsed = 0;
        perftVal = *globalPerftCounter;
    }
    else
    {
        // recurse serially till we reach a depth where we can launch parallel work
        uint64 newHash = hash;
        if (move != NULL)
        {
            newHash = makeMoveAndUpdateHash(pos, hash, *move, color);
            color = !color;
        }
        
        // look up the transposition table
#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[depth], 1);
#endif

        uint64 lookupHash = newHash ^ (ZOB_KEY(depth) * depth);

        TT_Entry entry = lookupTT(lookupHash, TTs->deepTT[depth], TTs->indexBits[depth]);
        if (searchTTEntry(entry, lookupHash, &perftVal, TTs->hashBits[depth]))
        {
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[depth], 1);
#endif
            return perftVal;
        }

        nMoves = generateMoves(pos, color, movesStack);
        *boardPtrStack = boardStack;
        for (uint32 i=0; i < nMoves; i++)
        {
            *boardStack = *pos;

            perftVal += perft_bb_gpu_hash_recursive_launcher(boardPtrStack, newHash, &movesStack[i], globalPerftCounter, depth-1, 
                                                            &movesStack[MAX_MOVES],  boardStack + 1, boardPtrStack + 1, launchDepth, TTs);
        }

        // store perft val in transposition table
#if PRINT_HASH_STATS == 1
        atomicAdd(&numStores[depth], 1);
#endif
        storeTTEntry(entry, lookupHash, depth, perftVal, TTs->deepTT[depth], TTs->indexBits[depth]);
    }

    return perftVal;
}

// the starting kernel for perft (which makes use of a hash table)
__global__ void perft_bb_driver_gpu_hash(HexaBitBoardPosition *pos, uint64 *globalPerftCounter, int depth, 
                                         void *serialStack, void *devMemory, int launchDepth, TTInfo TTs)
{
    // set device memory pointers
    preAllocatedBuffer = devMemory;
    preAllocatedMemoryUsed = 0;

    
    // call the recursive function
    // Three items are stored in the stack
    // 1. the board position pointer (one item per level)
    // 2. the board position         (one item per level)
    // 3. generated moves            (upto MAX_MOVES item per level)
    HexaBitBoardPosition *boardStack        = (HexaBitBoardPosition *)  serialStack;
    HexaBitBoardPosition **boardPtrStack    = (HexaBitBoardPosition **) ((char *)serialStack + (16 * 1024));
    CMove *movesStack                       = (CMove *)                 ((char *)serialStack + (20 * 1024));

    *boardPtrStack = pos;   // put the given board in the board ptr stack
    uint64 hash = MoveGeneratorBitboard::computeZobristKey(pos);
    //printf("\nOrignal board hash: %X%X\n", HI(hash), LO(hash));


#if PRINT_HASH_STATS == 1
    for (int i=0;i<MAX_GAME_LENGTH;i++)
    {
        numProbes[i] = 0;
        numHits[i] = 0;
        numStores[i] = 0;
    }
#endif

#if COUNT_NUM_COUNT_MOVES == 1
    numCountMoves = 0ull;
#endif
    uint64 finalPerfVal = perft_bb_gpu_hash_recursive_launcher(boardPtrStack, hash, NULL, globalPerftCounter, depth, movesStack, boardStack, 
                                                                boardPtrStack+1, launchDepth, &TTs);
#if COUNT_NUM_COUNT_MOVES == 1
    printf("Total no. of times countMoves was called: %llu \n", numCountMoves);
#endif

#if PRINT_HASH_STATS == 1
    printf("\nHash stats per depth\n");
    printf("depth   hash probes      hash hits    hash stores\n");
    for (int i=2; i<=depth; i++)
        printf("%5d   %11llu    %11llu    %11llu\n", i, numProbes[i], numHits[i], numStores[i]);
#endif

    *globalPerftCounter = finalPerfVal;
}





static void hugeMemset(void *data, uint64 size)
{
	uint8 *mem = (uint8*) data;
	const uint64 c4G = 4ull * 1024 * 1024 * 1024;

    while (size > c4G)
    {
        cudaMemset(mem, 0, c4G);
		
		mem  += c4G;
		size -= c4G;
    }
    
	cudaMemset(mem, 0, size);
}


// Transposition table pointers

// the shallow tables. 64 bit (8 byte) entries
uint64           *gTTDepth2_cpu;
uint64           *gTTDepth3_cpu;
uint64           *gTTDepth4_cpu;

// (one or more) deep tables (shared across multiple levels). Either 128 bit (16 byte) or 32 byte entries
TT_Entry         *gTranspositionTable_cpu;

// allocate hash tables for GPU perft
// TODO: free these tables!
void setupHashTables(TTInfo &TransTables)
{
    // allocate the hash tables
#if USE_SYSMEM_HASH == 1
    // try allocating in system memory
    void *temp;
    cudaError_t res = cudaHostAlloc(&temp, TT_SIZE * sizeof(TT_Entry), cudaHostAllocMapped | cudaHostAllocWriteCombined );
    if (res != S_OK)
    {
        printf("\nFailed to allocate GPU transposition table of %d bytes, with error: %s\n", TT_SIZE * sizeof(TT_Entry), cudaGetErrorString(res));
    }
    res = cudaHostGetDevicePointer(&gTranspositionTable_cpu, temp, 0);
    if (res != S_OK)
    {
        printf("\nFailed to get GPU mapping for sysmem hash table, with error: %s\n", cudaGetErrorString(res));
    }
#else
    cudaError_t res = cudaMalloc(&gTranspositionTable_cpu, TT_SIZE * sizeof(TT_Entry));
    if (res != S_OK)
    {
        printf("\nFailed to allocate GPU transposition table of %d bytes, with error: %s\n", TT_SIZE * sizeof(TT_Entry), cudaGetErrorString(res));
    }
#endif
	hugeMemset(gTranspositionTable_cpu, TT_SIZE * sizeof(TT_Entry));

    // first shallow TT (for storing depth 3 positions)
    res = cudaMalloc(&gTTDepth3_cpu, SHALLOW_TT3_SIZE * sizeof(uint64));
    if (res != S_OK)
    {
        printf("\nFailed to allocate GPU depth3 transposition table of %d bytes, with error: %s\n", SHALLOW_TT3_SIZE * sizeof(uint64), cudaGetErrorString(res));
    }
    hugeMemset(gTTDepth3_cpu, SHALLOW_TT3_SIZE * sizeof(uint64));

    // second shallow transposition table (for storing depth 2 positions)
    res = cudaMalloc(&gTTDepth2_cpu, SHALLOW_TT2_SIZE * sizeof(uint64));
    if (res != S_OK)
    {
        printf("\nFailed to allocate GPU depth2 transposition table of %d bytes, with error: %s\n", SHALLOW_TT2_SIZE * sizeof(uint64), cudaGetErrorString(res));
    }
    hugeMemset(gTTDepth2_cpu, SHALLOW_TT2_SIZE * sizeof(uint64));

	// third shallow transposition table (for storing depth 4 positions)
    res = cudaMalloc(&gTTDepth4_cpu, SHALLOW_TT4_SIZE * sizeof(uint64));
    if (res != S_OK)
    {
        printf("\nFailed to allocate GPU depth4 transposition table of %d bytes, with error: %s\n", SHALLOW_TT4_SIZE * sizeof(uint64), cudaGetErrorString(res));
    }
    hugeMemset(gTTDepth4_cpu, SHALLOW_TT4_SIZE * sizeof(uint64));
		


    // set pointers to the appriopriate hash tables for each depth
    for (int i=0;i<16;i++)
    {
        TransTables.deepTT[i] = gTranspositionTable_cpu;
        TransTables.indexBits[i] = TT_INDEX_BITS;
        TransTables.hashBits[i] = TT_HASH_BITS;
    }

    TransTables.depth2 = gTTDepth2_cpu;
    TransTables.depth3 = gTTDepth3_cpu;
    TransTables.depth4 = gTTDepth4_cpu;
    TransTables.indexBits[2] = SHALLOW_TT2_INDEX_BITS;
    TransTables.indexBits[3] = SHALLOW_TT3_INDEX_BITS;
    TransTables.indexBits[4] = SHALLOW_TT4_INDEX_BITS;
    TransTables.hashBits[2] = SHALLOW_TT2_HASH_BITS;
    TransTables.hashBits[3] = SHALLOW_TT3_HASH_BITS;
    TransTables.hashBits[4] = SHALLOW_TT4_HASH_BITS;
}
#endif