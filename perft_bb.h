// functions for computing perft using bitboard board representation

#include <cub/cub.cuh>

// the routines that actually generate the moves
#include "MoveGeneratorBitboard.h"

// preallocated GPU memory buffer (host-side pointer)
void *preAllocatedBufferHost;

// helper routines for CPU perft
uint32 countMoves(QuadBitBoard *pos, GameState *gs)
{
    uint32 nMoves;
    int chance = gs->chance;

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        nMoves = MoveGeneratorBitboard::countMoves<BLACK>(pos, gs);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::countMoves<WHITE>(pos, gs);
    }
#else
    nMoves = MoveGeneratorBitboard::countMoves(pos, gs, chance);
#endif
    return nMoves;
}

__host__ __device__ __forceinline__ uint32 generateMoves(QuadBitBoard *pos, GameState *gs, uint8 color, CMove *genMoves)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateMoves<BLACK>(pos, gs, genMoves);
    }
    else
    {
        return MoveGeneratorBitboard::generateMoves<WHITE>(pos, gs, genMoves);
    }
#else
    return MoveGeneratorBitboard::generateMoves(pos, gs, genMoves, color);
#endif
}

// A very simple CPU routine - only for estimating launch depth
uint64 perft_bb(QuadBitBoard *pos, GameState *gs, uint32 depth)
{
    if (depth == 1)
    {
        return countMoves(pos, gs);
    }

    CMove moves[MAX_MOVES];
    int nMoves = generateMoves(pos, gs, gs->chance, moves);

    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        QuadBitBoard childPos = *pos;
        GameState childGs = *gs;
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (gs->chance == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE>(&childPos, &childGs, moves[i]);
        else
            MoveGeneratorBitboard::makeMove<BLACK>(&childPos, &childGs, moves[i]);
#else
        MoveGeneratorBitboard::makeMove(&childPos, &childGs, moves[i], gs->chance);
#endif
        count += perft_bb(&childPos, &childGs, depth - 1);
    }
    return count;
}


// fixed
#define WARP_SIZE 32

#define ALIGN_UP(addr, align)   (((addr) + (align) - 1) & (~((align) - 1)))
#define MEM_ALIGNMENT 16

// makes the given move on the given position
__device__ __forceinline__ void makeMove(QuadBitBoard *pos, GameState *gs, CMove move, int chance)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(pos, gs, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(pos, gs, move);
    }
#else
    MoveGeneratorBitboard::makeMove(pos, gs, move, chance);
#endif
}

__host__ __device__ __forceinline__ uint32 countMoves(QuadBitBoard *pos, GameState *gs, uint8 color)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::countMoves<BLACK>(pos, gs);
    }
    else
    {
        return MoveGeneratorBitboard::countMoves<WHITE>(pos, gs);
    }
#else
    return MoveGeneratorBitboard::countMoves(pos, gs, color);
#endif
}


// fast reduction for the warp
__device__ __forceinline__ void warpReduce(int &x)
{
    #pragma unroll
    for(int mask = 16; mask > 0 ; mask >>= 1)
        x += __shfl_xor_sync(0xFFFFFFFF, x, mask);
}

// -------------------------------------------------------------------------
// Merge-path interval expand kernels
// -------------------------------------------------------------------------

#define IE_VT 7
#define IE_NV (BLOCK_SIZE * IE_VT)

// Phase 1: Compute merge-path partition points.
__global__ void merge_path_partitions(
    const int *inclPrefixSums,
    int aCount,
    int bCount,
    int numPartitions,
    int *partitions)
{
    int partition = blockIdx.x * blockDim.x + threadIdx.x;
    if (partition >= numPartitions)
        return;

    long long diag_ll = (long long)partition * IE_NV;
    long long total_ll = (long long)aCount + bCount;
    int diag = (int)(diag_ll < total_ll ? diag_ll : total_ll);

    int lo = (diag - bCount) > 0 ? (diag - bCount) : 0;
    int hi = diag < aCount ? diag : aCount;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        int b_idx = diag - 1 - mid;
        if (b_idx < 0 || mid < inclPrefixSums[b_idx])
            lo = mid + 1;
        else
            hi = mid;
    }
    partitions[partition] = lo;
}

// Phase 2: Merge-path interval expand kernel.
__global__ void mp_interval_expand_kernel(
    int *output,
    const int *inclPrefixSums,
    int aCount,
    int bCount,
    const int *partitions)
{
    __shared__ int shared[BLOCK_SIZE * (IE_VT + 1)];

    int block = blockIdx.x;
    int tid = threadIdx.x;

    int a0 = partitions[block];
    int a1 = partitions[block + 1];
    long long d0_ll = (long long)block * IE_NV;
    long long d1_ll = (long long)(block + 1) * IE_NV;
    long long total_ll = (long long)aCount + bCount;
    int diag0 = (int)(d0_ll < total_ll ? d0_ll : total_ll);
    int diag1 = (int)(d1_ll < total_ll ? d1_ll : total_ll);
    int b0 = diag0 - a0;
    int b1 = diag1 - a1;

    int aCount_cta = a1 - a0;
    int bCount_cta = b1 - b0;

    int *s_output = shared;
    int *s_prefix = shared + aCount_cta;

    for (int i = tid; i < bCount_cta; i += BLOCK_SIZE)
        s_prefix[i] = inclPrefixSums[b0 + i];
    __syncthreads();

    int merge_total = aCount_cta + bCount_cta;
    int diag_local_raw = IE_VT * tid;
    int diag_local = diag_local_raw < merge_total ? diag_local_raw : merge_total;
    int lo = (diag_local - bCount_cta) > 0 ? (diag_local - bCount_cta) : 0;
    int hi = diag_local < aCount_cta ? diag_local : aCount_cta;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        int b_idx = diag_local - 1 - mid;
        if (b_idx < 0 || (a0 + mid) < s_prefix[b_idx])
            lo = mid + 1;
        else
            hi = mid;
    }

    int a_cur = lo;
    int b_cur = diag_local - a_cur;
    int b_val = (b_cur < bCount_cta) ? s_prefix[b_cur] : INT_MAX;

    #pragma unroll
    for (int i = 0; i < IE_VT; i++)
    {
        bool consumeA = (a_cur < aCount_cta) &&
                         ((b_cur >= bCount_cta) || ((a0 + a_cur) < b_val));
        if (consumeA)
        {
            s_output[a_cur] = b0 + b_cur;
            a_cur++;
        }
        else if (b_cur < bCount_cta)
        {
            b_cur++;
            b_val = (b_cur < bCount_cta) ? s_prefix[b_cur] : INT_MAX;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < IE_VT; i++)
    {
        int index = BLOCK_SIZE * i + tid;
        if (index < aCount_cta)
        {
            output[a0 + index] = s_output[index];
        }
    }
}

// Host-side memory allocator from preallocated GPU buffer
struct GpuBumpAllocator
{
    void *base;
    size_t offset;
    size_t capacity;

    GpuBumpAllocator(void *buf, size_t cap) : base(buf), offset(0), capacity(cap) {}

    template<typename T>
    T* alloc(size_t count)
    {
        size_t bytes = count * sizeof(T);
        bytes = ALIGN_UP(bytes, MEM_ALIGNMENT);
        if (offset + bytes > capacity)
        {
            printf("\nOOM: need %llu bytes, only %llu available\n", (unsigned long long)(offset + bytes), (unsigned long long)capacity);
            return nullptr;
        }
        T* ptr = (T*)((uint8*)base + offset);
        offset += bytes;
        return ptr;
    }

    void reset() { offset = 0; }
};

// -------------------------------------------------------------------------
// GPU kernels for the host-driven BFS perft
// -------------------------------------------------------------------------

// Kernel: make move on parent board, produce child board, count child moves
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makemove_and_count_moves_kernel(QuadBitBoard *parentBoards, GameState *parentStates,
                                                 int *indices, CMove *moves,
                                                 QuadBitBoard *outPositions, GameState *outStates,
                                                 int *moveCounts, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    int nMoves = 0;

    if (index < nThreads)
    {
        int parentIndex = indices[index];
        QuadBitBoard pos = parentBoards[parentIndex];
        GameState gs = parentStates[parentIndex];
        CMove move = moves[index];

        uint8 color = gs.chance;
        makeMove(&pos, &gs, move, color);
        nMoves = countMoves(&pos, &gs, !color);

        outPositions[index] = pos;
        outStates[index] = gs;
        moveCounts[index] = nMoves;
    }
}

// Kernel: generate moves for each position (uses inclusive prefix sums for offsets)
__global__ void generate_moves_kernel(QuadBitBoard *positions, GameState *states,
                                       CMove *generatedMovesBase, int *inclPrefixSums, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nThreads)
    {
        QuadBitBoard pos = positions[index];
        GameState gs = states[index];
        int offset = (index == 0) ? 0 : inclPrefixSums[index - 1];
        CMove *genMoves = generatedMovesBase + offset;
        uint8 color = gs.chance;
        generateMoves(&pos, &gs, color, genMoves);
    }
}

// Kernel: leaf level - make move and count, add to global perft counter
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makeMove_and_perft_leaf_kernel(QuadBitBoard *positions, GameState *states,
                                                int *indices, CMove *moves,
                                                uint64 *globalPerftCounter, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    uint32 boardIndex = indices[index];
    QuadBitBoard pos = positions[boardIndex];
    GameState gs = states[boardIndex];
    int color = gs.chance;

    CMove move = moves[index];

    makeMove(&pos, &gs, move, color);

    // count moves at this position
    int nMoves = countMoves(&pos, &gs, !color);

    // warp-wide reduction before atomic add
    warpReduce(nMoves);

    int laneId = threadIdx.x & 0x1f;

    if (laneId == 0)
    {
        atomicAdd(globalPerftCounter, nMoves);
    }
}

// Kernel: interval expand using inclusive prefix sums (naive binary search)
__global__ void interval_expand_kernel(int *output, const int *inclPrefixSums, int numParents, int numOutput)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numOutput)
        return;

    int lo = 0, hi = numParents;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (inclPrefixSums[mid] <= tid)
            lo = mid + 1;
        else
            hi = mid;
    }
    output[tid] = lo;
}


// -------------------------------------------------------------------------
// Host-driven BFS perft
// -------------------------------------------------------------------------

// Run perft on GPU for a single position using host-driven breadth-first search
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, int depth, void *gpuBuffer, size_t bufferSize)
{
    if (depth <= 0)
        return 1;
    if (depth == 1)
        return countMoves(pos, gs);

    GpuBumpAllocator alloc(gpuBuffer, bufferSize);

    // Pre-allocate CUB temp storage once
    void *d_cubTemp = nullptr;
    size_t cubTempBytes = 0;
    cub::DeviceScan::InclusiveSum(d_cubTemp, cubTempBytes, (int*)nullptr, (int*)nullptr, 256 * 1024 * 1024);
    d_cubTemp = alloc.alloc<uint8>(cubTempBytes);

    // Copy root position to GPU (separate arrays)
    QuadBitBoard *d_rootPos = alloc.alloc<QuadBitBoard>(1);
    GameState *d_rootState = alloc.alloc<GameState>(1);
    cudaMemcpy(d_rootPos, pos, sizeof(QuadBitBoard), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rootState, gs, sizeof(GameState), cudaMemcpyHostToDevice);

    // Generate root moves on CPU
    CMove rootMoves[MAX_MOVES];
    int rootMoveCount = generateMoves(pos, gs, gs->chance, rootMoves);

    if (rootMoveCount == 0)
        return 0;

    // Copy root moves to GPU
    CMove *d_moves = alloc.alloc<CMove>(rootMoveCount);
    cudaMemcpy(d_moves, rootMoves, sizeof(CMove) * rootMoveCount, cudaMemcpyHostToDevice);

    // Create root indices (all point to index 0 = the root position)
    int *d_indices = alloc.alloc<int>(rootMoveCount);
    cudaMemset(d_indices, 0, sizeof(int) * rootMoveCount);

    // Perft counter on GPU
    uint64 *d_perftCounter = alloc.alloc<uint64>(1);
    cudaMemset(d_perftCounter, 0, sizeof(uint64));

    // Pre-allocate partition array for merge-path interval expand
    const int maxPartitions = 500000;
    int *d_partitions = alloc.alloc<int>(maxPartitions);

    QuadBitBoard *d_prevBoards = d_rootPos;
    GameState *d_prevStates = d_rootState;
    int currentCount = rootMoveCount;

    // BFS loop: depth-1 down to 2 are intermediate levels, level 1 is the leaf
    for (int level = depth - 1; level >= 2; level--)
    {
        // Allocate boards, states, and move counts
        QuadBitBoard *d_curBoards = alloc.alloc<QuadBitBoard>(currentCount);
        GameState *d_curStates = alloc.alloc<GameState>(currentCount);
        int *d_moveCounts = alloc.alloc<int>(currentCount);

        if (!d_curBoards || !d_curStates || !d_moveCounts)
        {
            printf("\nOOM during BFS at level %d with %d positions\n", level, currentCount);
            return 0;
        }

        // Step 1: make moves and count child moves
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        makemove_and_count_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                   d_indices, d_moves,
                                                                   d_curBoards, d_curStates,
                                                                   d_moveCounts, currentCount);

        // Step 2: in-place inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_cubTemp, cubTempBytes, d_moveCounts, d_moveCounts, currentCount);

        // Step 3: read back total
        int nextLevelCount = 0;
        cudaMemcpy(&nextLevelCount, d_moveCounts + currentCount - 1, sizeof(int), cudaMemcpyDeviceToHost);

        if (nextLevelCount == 0)
            break;

        // Allocate next level arrays
        int *d_newIndices = alloc.alloc<int>(nextLevelCount);
        CMove *d_newMoves = alloc.alloc<CMove>(nextLevelCount);

        if (!d_newIndices || !d_newMoves)
        {
            printf("\nOOM for next level with %d moves\n", nextLevelCount);
            return 0;
        }

        // Step 4: merge-path interval expand
        int mergeItems = nextLevelCount + currentCount;
        int numExpandCTAs = (mergeItems + IE_NV - 1) / IE_NV;
        int numPartitions = numExpandCTAs + 1;

        if (numPartitions > maxPartitions)
        {
            printf("\nToo many partitions (%d) for pre-allocated buffer (%d)\n", numPartitions, maxPartitions);
            return 0;
        }

        int partBlocks = (numPartitions + 127) / 128;
        merge_path_partitions<<<partBlocks, 128>>>(d_moveCounts, nextLevelCount, currentCount,
                                                    numPartitions, d_partitions);

        mp_interval_expand_kernel<<<numExpandCTAs, BLOCK_SIZE>>>(d_newIndices, d_moveCounts,
                                                                   nextLevelCount, currentCount, d_partitions);

        // Step 5: generate moves
        generate_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_curBoards, d_curStates, d_newMoves, d_moveCounts, currentCount);

        // Advance to next level
        d_prevBoards = d_curBoards;
        d_prevStates = d_curStates;
        d_indices = d_newIndices;
        d_moves = d_newMoves;
        currentCount = nextLevelCount;
    }

    // Final level: make move and count (leaf)
    {
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        makeMove_and_perft_leaf_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                  d_indices, d_moves,
                                                                  d_perftCounter, currentCount);
        cudaDeviceSynchronize();
    }

    // Read back result
    uint64 result = 0;
    cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);

    return result;
}
