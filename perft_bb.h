// functions for computing perft using bitboard board representation

// the routines that actually generate the moves
#include "MoveGeneratorBitboard.h"

#define MAX_GPUs 4
           void   *preAllocatedBufferHost[MAX_GPUs];
__device__ void   *preAllocatedBuffer;
__device__ uint32  preAllocatedMemoryUsed;

// use parallel scan and interval expand algorithms (from modern gpu lib) for 
// performing the move list scan and 'expand' operation to set correct board pointers (of parent boards) for second level child moves

// Another possible idea to avoid this operation is to have GenerateMoves() generate another array containing the indices 
// of the parent boards that generated the move (i.e, the global thread index for generateMoves kernel)
// A scan will still be needed to figure out starting address to write, but we won't need the interval expand
#include "moderngpu-master/include/kernels/scan.cuh"
#include "moderngpu-master/include/kernels/intervalmove.cuh"

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
// 256 works best for Maxwell
// (also make sure max registers used is set to 47)
#define BLOCK_SIZE 256

// fixed
#define WARP_SIZE 32

#define ALIGN_UP(addr, align)   (((addr) + (align) - 1) & (~((align) - 1)))
#define MEM_ALIGNMENT 16

// set this to true if devicMalloc can be called from multiple threads
#define MULTI_THREADED_MALLOC 1

template<typename T>
__device__ __forceinline__ int deviceMalloc(T **ptr, uint32 size)
{
    // align up the size to nearest 16 bytes (as some structures might have assumed 16 byte alignment?)
    size = ALIGN_UP(size, MEM_ALIGNMENT);

#if	MULTI_THREADED_MALLOC == 1
    uint32 startOffset = atomicAdd(&preAllocatedMemoryUsed, size);
#else
	uint32 startOffset = preAllocatedMemoryUsed;
    preAllocatedMemoryUsed += size;
#endif

    
    if (startOffset >= PREALLOCATED_MEMORY_SIZE)
    {
        //printf("\nFailed allocating %d bytes\n", size);
        //return -1;
    }
    

    *ptr = (T*) ((uint8 *)preAllocatedBuffer + startOffset);

    //printf("\nAllocated %d bytes at address: %X\n", size, *ptr);

    return S_OK;
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
__device__ __forceinline__ HashKey128b makeMoveAndUpdateHash(HexaBitBoardPosition *pos, HashKey128b hash, CMove move, int chance)
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

__host__ __device__ __forceinline__ uint32 countMoves(HexaBitBoardPosition *pos, uint8 color)
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

__host__ __device__ __forceinline__ uint32 generateBoards(HexaBitBoardPosition *pos, uint8 color, HexaBitBoardPosition *childBoards)
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


__host__ __device__ __forceinline__ uint32 generateMoves(HexaBitBoardPosition *pos, uint8 color, CMove *genMoves)
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

#define MAX_PERFT_DEPTH 16

struct TTInfo128b
{
    // gpu pointers to the hash tables
    void *hashTable[MAX_PERFT_DEPTH];

    // cpu pointers to the hash tables
    void *cpuTable[MAX_PERFT_DEPTH];

    // mask of index and hash bits for each transposition table
    uint64 indexBits[MAX_PERFT_DEPTH];
    uint64 hashBits[MAX_PERFT_DEPTH];
    bool   shallowHash[MAX_PERFT_DEPTH];
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

// same as above function but works with indices
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, 4)
#endif
__global__ void makeMove_and_perft_single_level_indices(HexaBitBoardPosition *positions, int *indices, CMove *moves, uint64 *globalPerftCounter, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    uint32 boardIndex = indices[index];
    HexaBitBoardPosition pos = positions[boardIndex];
    int color = pos.chance;

    CMove move = moves[index];

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
        atomicAdd(globalPerftCounter, nMoves);
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

		moveCounts[index] = nMoves;
	}
}

// this kernel does several things
// 1. Figures out the parent board position using indices[] array to lookup in parentBoards[] array
// 2. makes the move on parent board to produce current board. Writes it to outPositions[].
// 3. Counts moves at current board position and writes it to moveCounts[].
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, 4)
#endif
__global__ void makemove_and_count_moves_single_level(HexaBitBoardPosition *parentBoards, int *indices, CMove *moves,
                                                      HexaBitBoardPosition *outPositions, int *moveCounts, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    // count the no. of moves for each board and save it in moveCounts array
    int nMoves = 0;

    if (index < nThreads)
    {
        int parentIndex = indices[index];
        HexaBitBoardPosition pos = parentBoards[parentIndex];
        CMove move = moves[index];

        //Utils::displayCompactMove(move);

        uint8 color = pos.chance;
        makeMove(&pos, move, color);
        nMoves = countMoves(&pos, !color);

        outPositions[index] = pos;
        moveCounts[index] = nMoves;
    }
}

#if 0
// serial recursive perft routine for testing
__device__ __host__ uint32 serial_perft_test(HexaBitBoardPosition *pos, int depth)
{
    uint8 color = pos->chance;
    uint32 nMoves = countMoves(pos, color);
    if (depth == 1)
        return nMoves;


    HexaBitBoardPosition *childPositions;
#ifdef __CUDA_ARCH__
    deviceMalloc(&childPositions, sizeof(HexaBitBoardPosition) * nMoves);
#else
    childPositions = (HexaBitBoardPosition*) malloc(sizeof(HexaBitBoardPosition)* nMoves);
#endif
    generateBoards(pos, color, childPositions);

    uint32 p = 0;
    for (int i = 0; i < nMoves; i++)
    {
        p += serial_perft_test(&(childPositions[i]), depth - 1);
    }
    return p;
}
#endif 

// this kernel does several things
// 1. Figures out the parent board position using indices[] array to lookup in parentBoards[] array
// 2. makes the move on parent board to produce current board. Writes it to outPositions[], also updates outHashes with new hash
// 3. looks up the transposition table to see if the current board is present, and if so, updates the perftCounter directly
// 4. which perft counter to update and the hash of parent board is also found by 
//    indexing using indices[] array into parentHashes[]/parentCounters[] arrays
// 5. clears the perftCountersCurrentDepth[] array passed in

// this should be called for 'shallow' depths - i.e, the ones where each TT entry is just 128 bytes (and perft value fits in 24 bits)
// use the next function for deeper depths (that need > 32 bit perft values)
template <typename PT, typename CT>
__global__ void makemove_and_count_moves_single_level_hash128b(HexaBitBoardPosition *parentBoards, HashKey128b *parentHashes,
                                                               PT *parentCounters, int *indices,  CMove *moves, 
                                                               ShallowHashEntry128b *hashTable, uint64 hashBits, uint64 indexBits,
                                                               HexaBitBoardPosition *outPositions, HashKey128b *outHashes,
                                                               int *moveCounts, CT *perftCountersCurrentDepth, 
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
        HashKey128b hash = parentHashes[parentIndex];
        PT *perftCounter = parentCounters + parentIndex;
        CMove move = moves[index];

        uint8 color = pos.chance;
        hash = makeMoveAndUpdateHash(&pos, hash, move, color);

#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[depth], 1);
#endif
        // check in transposition table
        HashKey128b entry = hashTable[hash.lowPart & indexBits].hashKey;
        entry.highPart = entry.highPart ^ entry.lowPart;
        if ((entry.highPart == hash.highPart) && ((entry.lowPart & hashBits) == (hash.lowPart & hashBits)))
        {
            uint32 perftFromHash = (entry.lowPart & indexBits);

#if 0
            // for testing
            if (serial_perft_test(&pos, depth) != perftFromHash)
            {
                printf("\nwrong perft found in hash!!!\n");
                perftFromHash = serial_perft_test(&pos, depth);
            }
#endif
            
            // hash hit
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[depth], 1);
#endif
            atomicAdd(perftCounter, perftFromHash);

            // mark it invalid so that no further work gets done on this board
            pos.whitePieces = 0;    // mark it invalid so that generatemoves doesn't generate moves
            hash.highPart = 0;      // mark it invalid so that PerftNFromNminus1 kernel ignores this
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

// same as above function - but using deep hash tables
__global__ void makemove_and_count_moves_single_level_hash128b_deep(HexaBitBoardPosition *parentBoards, HashKey128b *parentHashes,
                                                                    uint64 *parentCounters, int *indices,  CMove *moves, 
                                                                    HashEntryPerft128b *hashTable, uint64 hashBits, uint64 indexBits,
                                                                    HexaBitBoardPosition *outPositions, HashKey128b *outHashes,
                                                                    int *moveCounts, uint64 *perftCountersCurrentDepth, 
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
        HashKey128b hash = parentHashes[parentIndex];
        uint64 *perftCounter = parentCounters + parentIndex;
        CMove move = moves[index];

        uint8 color = pos.chance;
        hash = makeMoveAndUpdateHash(&pos, hash, move, color);

#if PRINT_HASH_STATS == 1
        atomicAdd(&numProbes[depth], 1);
#endif
        // check in transposition table
        HashEntryPerft128b entry = hashTable[hash.lowPart & indexBits];
        // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
        entry.hashKey.highPart ^= entry.perftVal;
        entry.hashKey.lowPart ^= entry.perftVal;
        if ((entry.hashKey.highPart == hash.highPart) && ((entry.hashKey.lowPart & hashBits) == (hash.lowPart & hashBits))
            && (entry.depth == depth))
        {
            // hash hit
#if PRINT_HASH_STATS == 1
            atomicAdd(&numHits[depth], 1);
#endif
            atomicAdd(perftCounter, entry.perftVal);

            // mark it invalid so that no further work gets done on this board
            pos.whitePieces = 0;    // mark it invalid so that generatemoves doesn't generate moves 
            hash.highPart = 0;      // mark it invalid so that perftNFromPerftNminus1 kernel doesn't process this
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

// For shallow depths only!
// 
// compute perft N from perft N-1 (using excessive atomic adds)
// also store perft (N-1) entry in the given hash table (hashes[] array is for positions at N-1 level)
// 'depth' is the value of (n-1)
template <typename PT>
__global__ void calcPerftNFromPerftNminus1_hash128b(PT *perftNCounters, int *indices,
                                                    uint32 *perftNminus1Counters, HashKey128b *hashes, HexaBitBoardPosition *boards,
                                                    HashKey128b *hashTable, uint64 hashBits, uint64 indexBits,
                                                    int nThreads, int depth)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nThreads)
    {
        HashKey128b hash = hashes[index];

        if (hash.highPart)  // hash == 0 means invalid entry - entry for which there was a hash hit
        {
            PT *perftNCounter = perftNCounters + indices[index];
            uint32 perftNminus1 = perftNminus1Counters[index];

            // get perft(N) from perft(N-1)
            // TODO: try replacing this atomic add with some parallel reduction trick (warp wide?)
            atomicAdd(perftNCounter, perftNminus1);

            //if (perftNminus1 > indexBits)
            //    printf("\nGot perft bigger than size in hash table\n");

            // store in hash table
            // it's assumed that perft value will fit in remaining (~hashMask) bits
            HashKey128b hashEntry = HashKey128b((hash.lowPart  & hashBits) | perftNminus1, hash.highPart);
            hashEntry.highPart ^= hashEntry.lowPart;
            hashTable[hash.lowPart & indexBits] = hashEntry;

#if PRINT_HASH_STATS == 1
            atomicAdd(&numStores[depth], 1);
#endif
        }
    }
}

// same as above kernel but for levels that need deep hash tables
__global__ void calcPerftNFromPerftNminus1_hash128b_deep(uint64 *perftNCounters, int *indices,
                                                         uint64 *perftNminus1Counters, HashKey128b *hashes,
                                                         HashEntryPerft128b *hashTable, uint64 hashBits, uint64 indexBits,
                                                         int nThreads, int depth)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nThreads)
    {
        HashKey128b hash = hashes[index];

        if (hash.highPart)  // hash == 0 means invalid entry - entry for which there was a hash hit
        {
            uint64 *perftNCounter = perftNCounters + indices[index];
            uint32 perftNminus1 = perftNminus1Counters[index];

            // get perft(N) from perft(N-1)
            // TODO: try replacing this atomic add with some parallel reduction trick (warp wide?)
            atomicAdd(perftNCounter, perftNminus1);

            // store in hash table
            HashEntryPerft128b oldEntry = hashTable[hash.lowPart & indexBits];
            // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
            oldEntry.hashKey.highPart ^= oldEntry.perftVal;
            oldEntry.hashKey.lowPart ^= oldEntry.perftVal;
            // replace only if old entry was shallower (or of same depth)
            if (oldEntry.depth <= depth)
            {
                HashEntryPerft128b newEntry;
                newEntry.perftVal = perftNminus1;
                newEntry.hashKey.highPart = hash.highPart;
                newEntry.hashKey.lowPart = (hash.lowPart & hashBits);
                newEntry.depth = depth;

                // XOR hash part with data part for lockless hashing
                newEntry.hashKey.lowPart  ^= newEntry.perftVal;
                newEntry.hashKey.highPart ^= newEntry.perftVal;

                hashTable[hash.lowPart & indexBits] = newEntry;

#if PRINT_HASH_STATS == 1
                atomicAdd(&numStores[depth], 1);
#endif
            }
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


// a simpler gpu perft routine
// only a single depth of kernel call nesting
// assumes that enough memory would be available
__global__ void perft_bb_gpu_simple(HexaBitBoardPosition *pos, uint64 *globalPerftCounter, int depth, void *devMemory)
{
    //preAllocatedBuffer = devMemory;
    //preAllocatedMemoryUsed = 0;

    HexaBitBoardPosition   *prevLevelBoards = NULL;         // boards at previous level

    int                     currentLevelCount = 0;          // no of positions at current level
    int                     *moveListOffsets = NULL;         // offset into moveList
    int                     *moveCounts = NULL;              // moveCounts of current level boards
    HexaBitBoardPosition    *currentLevelBoards = NULL;      // boards at current level

    int                     nextLevelCount = 0;             // sum of moveCounts
    CMove                  *childMoves;                     // moves generated by current level boards


    // special case for first level (root)
    uint8 color = pos->chance;
    nextLevelCount = countMoves(pos, color);

    if (nextLevelCount == 0)
    {
        return;
    }

    deviceMalloc(&childMoves, nextLevelCount * sizeof (CMove));
    generateMoves(pos, color, childMoves);

    prevLevelBoards = pos;
    currentLevelCount = nextLevelCount;

    deviceMalloc(&moveListOffsets, currentLevelCount * sizeof(int));
    for (int i = 0; i < currentLevelCount; i++)
        moveListOffsets[i] = 0;


    int nBlocks;

    // cuda stream for launching child kernels
    cudaStream_t childStream;
    cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

    //printf("\nBefore loop, currentLevelCount: %d\n", currentLevelCount);

    for (int i = 1; i < depth - 1; i++)
    {
        // allocate memory for current level boards, and moveCounts
        deviceMalloc(&currentLevelBoards, currentLevelCount * sizeof (HexaBitBoardPosition));

        int size = sizeof(int)* currentLevelCount;
        size = (int)size * 1.2f + 256;
        // (add some scratch space needed by scan and intervalExpand functions)
        deviceMalloc(&moveCounts, size);

        // make moves to get current level boards and get next level counts
        nBlocks = (currentLevelCount - 1) / BLOCK_SIZE + 1;

        makemove_and_count_moves_single_level << <nBlocks, BLOCK_SIZE, 0, childStream >> >
            (prevLevelBoards, moveListOffsets, childMoves,
            currentLevelBoards, moveCounts, currentLevelCount);

        // do a scan to get new moveListOffsets
        int *pNumSecondLevelMoves = moveCounts + currentLevelCount;     // global memory to hold the sum
        int *scratchSpace = pNumSecondLevelMoves + 1;

        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (moveCounts, currentLevelCount, moveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(),
            pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();

        // the scan also gives the total of moveCounts
        nextLevelCount = *pNumSecondLevelMoves;

        if (nextLevelCount == 0)
        {
            // unlikely, but possible
            return;
        }

        // Allocate memory for:
        // next level child moves (i.e, moves at child boards of current board positions)
        deviceMalloc(&childMoves, sizeof(CMove)* nextLevelCount);

        // and  indices to current level boards
        deviceMalloc(&moveListOffsets, sizeof(int)* nextLevelCount);

        // Generate secondToFirstLevelIndices by running interval expand
        // 
        // Expand currentLevelMoves items -> nextLevelCount items
        // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output
        // moveCounts now have the exclusive scan - containing the indices to put moves on

        mgpu::IntervalExpandDGenValues(nextLevelCount, moveCounts, (int) 0, 1, currentLevelCount, moveListOffsets, childStream, scratchSpace);

        // Generate next level child moves
        // moveCounts[] (containing exclusive scan) is used by the below kernel to index into childMoves[] - to know where to put the generated moves
        generate_moves_single_level << <nBlocks, BLOCK_SIZE, 0, childStream >> > (currentLevelBoards, childMoves, moveCounts, currentLevelCount);

        // go to next level
        currentLevelCount = nextLevelCount;
        prevLevelBoards = currentLevelBoards;

    }

    // special case for last level
    nBlocks = (currentLevelCount - 1) / BLOCK_SIZE + 1;
    makeMove_and_perft_single_level_indices << <nBlocks, BLOCK_SIZE, 0, childStream >> >
        (prevLevelBoards, moveListOffsets, childMoves, globalPerftCounter, currentLevelCount);

    cudaStreamDestroy(childStream);
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
        //perft_bb_gpu_safe<<<1, BLOCK_SIZE, sizeof(sharedMemAllocs), 0>>> (posPtr, move, globalPerftCounter, depth, 1);
        if (move != NULL)
        {
            makeMove(pos, move[0], color);
        }
        perft_bb_gpu_simple << <1, 1, 0, 0 >> > (pos, globalPerftCounter, depth, preAllocatedBuffer);
        cudaDeviceSynchronize();

        // 'free' up the memory used by the launch
        // printf("\nmemory used by previous parallel launch: %d bytes\n", preAllocatedMemoryUsed);
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


//--------------------------------------------------------------------------------------------------
// versions of the above kernel that use hash tables
//--------------------------------------------------------------------------------------------------

__global__ void setPreallocatedMemory(void *devMemory)
{
    preAllocatedBuffer = devMemory;
    preAllocatedMemoryUsed = 0;
}

// a simpler gpu perft routine (with hash table support)
// only a single depth of kernel call nesting
// assumes that enough memory would be available
// hashTables[] array is the array of transposition tables for each depth
// shallowHash[] array specifies if the hash table for the depth is a shallow hash table (128 bit hash entries - with value stored in index bits)
// indexBits[] and hashbits[] arrays are bitmasks for obtaining index and hash value from a 64 bit low part of hash value (for each depth)
// TODO: skip hash table check/store for depth 2 also?
__global__ void perft_bb_gpu_simple_hash(HexaBitBoardPosition *positions, HashKey128b *hashes, uint64 *perfts, int depth,
                                         void *devMemory, TTInfo128b ttInfo, bool newBatch)                                         
{
    void   **hashTables   = ttInfo.hashTable;
    bool   *shallowHash   = ttInfo.shallowHash;
    uint64 *indexBits     = ttInfo.indexBits;
    uint64 *hashBits      = ttInfo.hashBits;

    HexaBitBoardPosition *pos = &positions[threadIdx.x];
    uint64 *perftOut = &perfts[threadIdx.x];

    if (newBatch)
    {
        preAllocatedBuffer = devMemory;
        preAllocatedMemoryUsed = 0;
    }

    // High level algorithm - two passes:
    // downsweep :
    //  - traverse the tree from top to bottom (in breadth first manner) and generate all moves/boards
    //  - use the hash table(s) to cut down on work (no need to re-explore the sub-tree from a position if there is hash hit).
    // upsweep:
    //  - traverse the generated tree from bottom to top, and accumulate all sums (perft counters)
    //  - update hash table entries on the go

    HexaBitBoardPosition   *prevLevelBoards = NULL;         // boards at previous level
    HashKey128b            *prevLevelHashes = NULL;         // hashes of the above boards
    void                   *prevLevelPerftCounters = NULL;  // perft counts for the above boards (uint32 for shallow depths, 64 bit otherwise)

    int                     currentLevelCount = 0;           // no of positions at current level
    int                     *moveListOffsets = NULL;         // offset into moveList
    int                     *moveCounts = NULL;              // moveCounts of current level boards
    HexaBitBoardPosition    *currentLevelBoards = NULL;      // boards at current level
    HashKey128b             *currentLevelHashes = NULL;      // hashes of the above boards
    void                    *currentLevelPerftCounters = NULL; // perft counters of the above boards

    int                     nextLevelCount = 0;             // sum of moveCounts
    CMove                  *childMoves;                     // moves generated by current level boards

    // the tree: created/saved during downsweep pass, used during up-sweep pass
    int                     levelCounts[MAX_PERFT_DEPTH];
    void                   *perftCounters[MAX_PERFT_DEPTH];
    HashKey128b            *boardHashes[MAX_PERFT_DEPTH];
    int                    *parentIndices[MAX_PERFT_DEPTH];     // moveListOffsets : index of parent board

    // only needed for debugging!
    HexaBitBoardPosition   *boards[MAX_PERFT_DEPTH];

    // special case for first level (root)
    uint8 color = pos->chance;
    nextLevelCount = countMoves(pos, color);

    if (nextLevelCount == 0)
    {
        // it should be already initialized to 0
        //*perftOut = 0;
        return;
    }

    deviceMalloc(&childMoves, nextLevelCount * sizeof (CMove));
    generateMoves(pos, color, childMoves);

    prevLevelBoards = pos;
    prevLevelPerftCounters = perftOut;

    prevLevelHashes = &hashes[threadIdx.x];

    currentLevelCount = nextLevelCount;

    deviceMalloc(&moveListOffsets, currentLevelCount * sizeof(int));
    
    for (int i = 0; i < currentLevelCount; i++)
        moveListOffsets[i] = 0;

    int nBlocks;

    // cuda stream for launching child kernels
    cudaStream_t childStream;
    cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

    levelCounts[depth] = 1;
    perftCounters[depth] = perftOut;
    boardHashes[depth] = prevLevelHashes;
    parentIndices[depth] = NULL;     // no-parent, this is the root
    boards[depth] = pos;
    
    int curDepth = 0;

    for (curDepth = depth - 1; curDepth > 1; curDepth--)
    {
        // allocate memory for current level boards, and moveCounts
        deviceMalloc(&currentLevelBoards, currentLevelCount * sizeof (HexaBitBoardPosition));

        int size = sizeof(int)* currentLevelCount;
        size = (int)size * 1.2f + 256;      // (add some scratch space needed by scan and intervalExpand functions)

        deviceMalloc(&moveCounts, size);

        // allocate memory for current level board hashes and perft counters
        deviceMalloc(&currentLevelHashes, currentLevelCount * sizeof (HashKey128b));

        if (shallowHash[curDepth])
            deviceMalloc(&currentLevelPerftCounters, currentLevelCount * sizeof (uint32));
        else
            deviceMalloc(&currentLevelPerftCounters, currentLevelCount * sizeof (uint64));

        // save pointers  for downsweep pass
        levelCounts[curDepth] = currentLevelCount;
        parentIndices[curDepth] = moveListOffsets;
        perftCounters[curDepth] = currentLevelPerftCounters;
        boardHashes[curDepth] = currentLevelHashes;
        boards[curDepth] = currentLevelBoards;

        // make moves to get current level boards and get next level counts
        nBlocks = (currentLevelCount - 1) / BLOCK_SIZE + 1;

        if (shallowHash[curDepth])
        {
            ShallowHashEntry128b *hashTable = (ShallowHashEntry128b *)(hashTables[curDepth]);
            if (shallowHash[curDepth + 1])
            {
                makemove_and_count_moves_single_level_hash128b <<<nBlocks, BLOCK_SIZE, 0, childStream >>>
                                                                (prevLevelBoards, prevLevelHashes, 
                                                                (uint32*) prevLevelPerftCounters, moveListOffsets, childMoves,
                                                                hashTable, hashBits[curDepth], indexBits[curDepth],
                                                                currentLevelBoards, currentLevelHashes, 
                                                                moveCounts, (uint32*) currentLevelPerftCounters,
                                                                currentLevelCount, curDepth);
            }
            else
            {
                makemove_and_count_moves_single_level_hash128b <<<nBlocks, BLOCK_SIZE, 0, childStream >>>
                                                                (prevLevelBoards, prevLevelHashes, 
                                                                (uint64*) prevLevelPerftCounters, moveListOffsets, childMoves,
                                                                hashTable, hashBits[curDepth], indexBits[curDepth],
                                                                currentLevelBoards, currentLevelHashes, 
                                                                moveCounts, (uint32*) currentLevelPerftCounters,
                                                                currentLevelCount, curDepth);
            }
        }
        else
        {
            HashEntryPerft128b *hashTable = (HashEntryPerft128b *)(hashTables[curDepth]);
            // > 24 bit perft counters
            makemove_and_count_moves_single_level_hash128b_deep <<<nBlocks, BLOCK_SIZE, 0, childStream >>>
                                                                 (prevLevelBoards, prevLevelHashes, 
                                                                 (uint64*) prevLevelPerftCounters, moveListOffsets, childMoves,
                                                                 hashTable, hashBits[curDepth], indexBits[curDepth],
                                                                 currentLevelBoards, currentLevelHashes, 
                                                                 moveCounts, (uint64*)currentLevelPerftCounters,
                                                                 currentLevelCount, curDepth);
        }

        // do a scan to get new moveListOffsets
        int *pNumSecondLevelMoves = moveCounts + currentLevelCount;     // global memory to hold the sum
        int *scratchSpace = pNumSecondLevelMoves + 1;

        mgpu::ScanD<mgpu::MgpuScanTypeExc>
            (moveCounts, currentLevelCount, moveCounts, mgpu::ScanOp<mgpu::ScanOpTypeAdd, int>(),
            pNumSecondLevelMoves, false, childStream, scratchSpace);

        cudaDeviceSynchronize();

        // the scan also gives the total of moveCounts
        nextLevelCount = *pNumSecondLevelMoves;

        if (nextLevelCount == 0)
        {
            // unlikely, but possible
            break;
        }

        // Allocate memory for:
        // next level child moves (i.e, moves at child boards of current board positions)
        deviceMalloc(&childMoves, sizeof(CMove)* nextLevelCount);

        // and  indices to current level boards
        deviceMalloc(&moveListOffsets, sizeof(int)* nextLevelCount);

        // Generate secondToFirstLevelIndices by running interval expand
        // 
        // Expand currentLevelMoves items -> nextLevelCount items
        // The function takes a integer base number, takes an integer multiplier and performs integer addition to populate the output
        // moveCounts now have the exclusive scan - containing the indices to put moves on

        mgpu::IntervalExpandDGenValues(nextLevelCount, moveCounts, (int) 0, 1, currentLevelCount, moveListOffsets, childStream, scratchSpace);

        // Generate next level child moves
        // moveCounts[] (containing exclusive scan) is used by the below kernel to index into childMoves[] - to know where to put the generated moves
        generate_moves_single_level << <nBlocks, BLOCK_SIZE, 0, childStream >> > (currentLevelBoards, childMoves, moveCounts, currentLevelCount);

#if 0
        for (int i = 0; i < nextLevelCount; i++)
        {
            Utils::displayCompactMove(childMoves[i]);
        }
#endif

        // go to next level
        currentLevelCount = nextLevelCount;
        prevLevelBoards = currentLevelBoards;
        prevLevelPerftCounters = currentLevelPerftCounters;
        prevLevelHashes = currentLevelHashes;
    }

    // printf("\nMax Parallel work: %d threads\n", currentLevelCount);
    curDepth++;
    if (curDepth == 2)
    {
        // special case for last level (this should be the most expensive kernel launch by large margin)
        // prevLevelPerftCounters - is expected to be 32 bit here (containing perft2 values)
        nBlocks = (currentLevelCount - 1) / BLOCK_SIZE + 1;
        makeMove_and_perft_single_level_indices <<<nBlocks, BLOCK_SIZE, 0, childStream>>>
                                                 (prevLevelBoards, (uint32*) prevLevelPerftCounters,
                                                 moveListOffsets, childMoves, currentLevelCount);
    }

    // downsweep pass: propogate the perft values down to compute perft of root position
    for (; curDepth < depth; curDepth++)
    {
        // not needed as all child streams are launched serially?
        // cudaDeviceSynchronize();

        nBlocks = (levelCounts[curDepth] - 1) / BLOCK_SIZE + 1;
        if (shallowHash[curDepth])
        {
            HashKey128b *hashTable = (HashKey128b*)(hashTables[curDepth]);

            if (shallowHash[curDepth+1])
            {
                calcPerftNFromPerftNminus1_hash128b <<< nBlocks, BLOCK_SIZE, 0, childStream >>> 
                                                    ((uint32 *) perftCounters[curDepth + 1], parentIndices[curDepth],
                                                    (uint32 *)perftCounters[curDepth], boardHashes[curDepth], boards[curDepth],
                                                     hashTable, hashBits[curDepth], indexBits[curDepth], 
                                                     levelCounts[curDepth], curDepth);
            }
            else
            {
                calcPerftNFromPerftNminus1_hash128b <<< nBlocks, BLOCK_SIZE, 0, childStream >>> 
                                                    ((uint64 *) perftCounters[curDepth + 1], parentIndices[curDepth],
                                                    (uint32 *)perftCounters[curDepth], boardHashes[curDepth], boards[curDepth],
                                                     hashTable, hashBits[curDepth], indexBits[curDepth], 
                                                     levelCounts[curDepth], curDepth);
            }
        }
        else
        {
            HashEntryPerft128b *hashTable = (HashEntryPerft128b*)(hashTables[curDepth]);

            calcPerftNFromPerftNminus1_hash128b_deep <<< nBlocks, BLOCK_SIZE, 0, childStream >>> 
                                                    ((uint64 *) perftCounters[curDepth + 1], parentIndices[curDepth],
                                                     (uint64 *) perftCounters[curDepth], boardHashes[curDepth], /*boards[curDepth],*/
                                                     hashTable, hashBits[curDepth], indexBits[curDepth], 
                                                     levelCounts[curDepth], curDepth);                                                    
        }
    }

#if 0
    if (preAllocatedMemoryUsed > maxMemoryUsed)
    {
        maxMemoryUsed = preAllocatedMemoryUsed;
        printf("\nmemory used: %d\n", preAllocatedMemoryUsed);
    }
#endif
    cudaStreamDestroy(childStream);
}


__global__ void perft_bb_gpu_launcher_hash(HexaBitBoardPosition *pos, HashKey128b hash, uint64 *perftOut, int depth,
                                           void *devMemory, TTInfo128b ttInfo)
{
    preAllocatedBuffer = devMemory;
    preAllocatedMemoryUsed = 0;

    // generate moves for the current board and call the breadth first routine for all child boards
    uint8 color = pos->chance;
    int nMoves = countMoves(pos, color);

    CMove *moves;
    HexaBitBoardPosition *childBoards;
    HashKey128b *hashes;
    uint64 *perfts;
    deviceMalloc(&childBoards, sizeof(HexaBitBoardPosition)*nMoves);
    deviceMalloc(&moves, sizeof(CMove)*nMoves);
    deviceMalloc(&hashes, sizeof(HashKey128b)*nMoves);
    deviceMalloc(&perfts, sizeof(uint64)*nMoves);

    // base after reserving space for the above allocations
    uint32 base = preAllocatedMemoryUsed;

    generateMoves(pos, color, moves);

    HashEntryPerft128b *hashTable = (HashEntryPerft128b *)ttInfo.hashTable[depth - 1];
    uint64 indexBits = ttInfo.indexBits[depth - 1];
    uint64 hashBits = ttInfo.hashBits[depth - 1];

    int nNewBoards = 0;

    for (int i = 0; i < nMoves; i++)
    {
        childBoards[nNewBoards] = *pos;
        HashKey128b newHash = makeMoveAndUpdateHash(&childBoards[nNewBoards], hash, moves[i], color);

        // check in hash table
        HashEntryPerft128b entry;
        entry = hashTable[newHash.lowPart & indexBits];
        // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
        entry.hashKey.highPart ^= entry.perftVal;
        entry.hashKey.lowPart ^= entry.perftVal;

        if ((entry.hashKey.highPart == newHash.highPart) && ((entry.hashKey.lowPart & hashBits) == (newHash.lowPart & hashBits))
            && (entry.depth == (depth - 1)))
        {
            // hash hit
            (*perftOut) += entry.perftVal;
            continue;
        }

        hashes[nNewBoards] = newHash;
        perfts[nNewBoards] = 0;
        nNewBoards++;
    }

    // parallel 4 launches a time
    // slightly slower on GM107, much faster on GP104
#define NUM_PARALLEL 4
    for (int i = 0; i < nNewBoards; i += NUM_PARALLEL)
    {
        int threads = (nNewBoards - i) < NUM_PARALLEL ? (nNewBoards - i) : NUM_PARALLEL;
        perft_bb_gpu_simple_hash<<<1, threads, 0, 0>>> (&childBoards[i], &hashes[i], &perfts[i], depth - 1, NULL, ttInfo, false);
        cudaDeviceSynchronize();
        preAllocatedMemoryUsed = base;
    }

    // collect perft results and update hash table
    for (int i = 0; i < nNewBoards; i ++)
    {
        (*perftOut) += perfts[i];

        HashKey128b posHash128b = hashes[i];

        HashEntryPerft128b oldEntry;
        oldEntry = hashTable[posHash128b.lowPart & indexBits];

        // replace only if old entry was shallower (or of same depth)
        if (hashTable && oldEntry.depth <= (depth - 1))
        {
            HashEntryPerft128b newEntry;
            newEntry.perftVal = perfts[i];
            newEntry.hashKey.highPart = posHash128b.highPart;
            newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
            newEntry.depth = (depth - 1);

            // XOR hash part with data part for lockless hashing
            newEntry.hashKey.lowPart ^= newEntry.perftVal;
            newEntry.hashKey.highPart ^= newEntry.perftVal;

            hashTable[posHash128b.lowPart & indexBits] = newEntry;
        }

    }

    // TRY:
    //  1. calling perft_bb_gpu_simple_hash with threads per block = 4 (and a single block).
    //     - modify the function to pick pos from the array based on threadId.x and make hash a pointer!
    //     - done: ~15-20% gains in performance on GP104!
    // 
    //  2. hash table in this level
    //     - done
    //
    //  3. find duplicates in the depth first search using simple hash table mechanism.
    //      - kernel1: every thread atomically updates hash table location with it's hash + threadId
    //      - kernel2: every thread reads the hash table again to figure out if it got to update the hash table or somebody else
    //                 - if somebody else with different hash => not a duplicate
    //                 - if someone with same hash, but different threadId => duplicate with original result at the threadId => remove.
    // 
    // 4. check max memory used after duplicates are removed, and maybe increase GPU parallel work to 7 levels?
}
