// this file contains the various compile time settings/swithes

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif


// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 0

// 768 MB preallocated memory size (for holding the perft tree in GPU memory)
// on systems with more video memory (like Titan X), we can use 3x of this to hold bigger trees
//#define PREALLOCATED_MEMORY_SIZE (768 * 1024 * 1024ull)
#define PREALLOCATED_MEMORY_SIZE (2800 * 1024 * 1024ull)

// 512 KB ought to be enough for holding the stack for the serial part of the gpu perft
#define GPU_SERIAL_PERFT_STACK_SIZE (512 * 1024)

// use constant memory for accessing lookup tables (except for magic tables as they are huge)
// the default is to use texture cache via __ldg instruction
// (doesn't really affect performance either way. Maybe just a tiny bit slower with fancy magics)
// Ankan - improves performance on Maxwell a LOT!
#define USE_CONSTANT_MEMORY_FOR_LUT 1

// flag en-passent capture only when it's possible (in next move)
// default is to flag en-passent on every double pawn push
// This switch works only using using makeMove()
// This helps A LOT when using transposition tables (> 50% improvement in perft(12) time)!
#define EXACT_EN_PASSENT_FLAGGING 1

// make use of a hash table to avoid duplicate calculations due to transpositions
#define USE_TRANSPOSITION_TABLE 1

#if USE_TRANSPOSITION_TABLE == 1

// find duplicates in the each level of the parallel breadth first search
// and make sure we explore them only once
#define FIND_DUPLICATES_IN_BFS 1

// windows 64 bit vs 32 bit vs linux 64 bit compromise :-/
// Windows allows overclocking (gives about 10% extra performance)
// Windows 32 bit build is 7% faster than windows 64 bit build (for some unknown reason??)!
// .. but 32 bit windows build is address space limited.. best settings: don't use sysmem hash, TT_BITS: 25, TT2_BITS: 25, TT3_BITS: 26, TT4_BITS: 25 (total 1.5 GB HASH)
// .. 64 bit windows doesn't allow creating pow-2 vidmem allocations bigger than 1 GB, and sysmem allocations bigger than 2 GB. best settings: TT_BITS: 27, TT2/TT3 BITS: 26, TT4_BITS: 27 (total 4 GB HASH using 2 GB sysmem)
// .. Linux allows using any amount of memory (limited by sysmem/vidmem sizes), so best setting is to distribute all of available vidmem and sysmem to cover all transposition tables adequately
// 9 JUL 2016: Looks like on Windows 10 and with latest drivers, there is no longer any limitation. 
//  .. except for cudaHostAlloc() which doesn't seem to allow allocating > 6GB of memory on 16 GB system :-/

// size of the transposition table (in number of entries)
// for entries of 16 bytes (shallow TT entries)
// 28 bits: 256 million entries -> 4 GB
// 27 bits: 128 million entries -> 2 GB
// 25 bits: 32 million entries  -> 512 MB
#define TT_SIZE_FROM_BITS   (1ull << TT_BITS)

// size is simply 2 power of bits
#define GET_TT_SIZE_FROM_BITS(bits)   (1ull << (bits))

// bits of the hash used as index into the transposition table
#define GET_TT_INDEX_BITS(bits)       (GET_TT_SIZE_FROM_BITS(bits) - 1)

// remaining bits (that are stored per hash entry)
#define GET_TT_HASH_BITS(bits)        (ALLSET ^ GET_TT_INDEX_BITS(bits))

#endif

// count the no of times countMoves() got called (useful to find hash table effectiveness)
#define COUNT_NUM_COUNT_MOVES 0

// print  various hash statistics
#define PRINT_HASH_STATS 0

// move generation functions templated on chance
// +9% benefit on GM204
#define USE_TEMPLATE_CHANCE_OPT 1

// bitwise magic instead of if/else for castle flag updation
// turning this off helps Maxwell a little
#define USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION 0

// intel core 2 doesn't have popcnt instruction
#define USE_POPCNT 1

// pentium 4 doesn't have fast HW bitscan
#define USE_HW_BITSCAN 1

// use lookup tabls for figuring out squares in line and squares in between
#define USE_IN_BETWEEN_LUT 1
    
// use lookup table for king moves
#define USE_KING_LUT 1

// use lookup table for knight moves
#define USE_KNIGHT_LUT 1

// use lookup table (magics) for sliding moves
// reduces performance by ~7% for GPU version
// helps maxwell a lot (> +10%)
#define USE_SLIDING_LUT 1

// use fancy fixed-shift version - ~ 800 KB lookup tables
// (setting this to 0 enables plain magics - with 2.3 MB lookup table)
// plain magics is a bit faster at least for perft (on core 2 duo)
// fancy magics is clearly faster on more recent processors (ivy bridge)
// fancy magics a very very tiny bit faster for Maxwell (actually almost exactly same speed)
#define USE_FANCY_MAGICS 1

// use byte lookup for fancy magics (~150 KB lookup tables)
// around 3% slower than fixed shift fancy magics on CPU
// and > 10% slower on GPU!
#define USE_BYTE_LOOKUP_FANCY 0


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif
