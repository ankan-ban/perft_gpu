// this file contains the various compile time settings/swithes

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif


// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 0

// don't call cudaMalloc/cudaFree from device code, 
// suballocate from a pre-allocated buffer instead
// (most of the routines now won't work when this is OFF - i.e, we now rely on this)
#define USE_PREALLOCATED_MEMORY 1

// 768 MB ... to keep space for transposition tables
// just hope that this would be sufficient :'(

// Keeping 768 MB as preallocated memory size allows us to use 1.5 GBs hash table 
// and allows setting cudaLimitDevRuntimeSyncDepth to a decent depth
// use 1536 MB when not using hash tables (so that we can fit wider levels in a single launch)
#define PREALLOCATED_MEMORY_SIZE (1 * 768 * 1024 * 1024)

// 512 KB ought to be enough for holding the stack for the serial part of the gpu perft
#define GPU_SERIAL_PERFT_STACK_SIZE (512 * 1024)

// use constant memory for accessing lookup tables (except for magic tables as they are huge)
// the default is to use texture cache via __ldg instruction
// (doesn't really affect performance either way. Maybe just a tiny bit slower with fancy magics)
// Ankan - improves performance on Maxwell a LOT!
#define USE_CONSTANT_MEMORY_FOR_LUT 1

// use parallel scan and interval expand algorithms (from modern gpu lib) for 
// performing the move list scan and 'expand' operation to set correct board pointers for second level child moves

// Another possible idea to avoid this operation is to have GenerateMoves() generate another array containing the indices 
// of the parent boards that generated the move (i.e, the global thread index for generateMoves kernel)
// A scan will still be needed to figure out starting address to write, but we won't need the interval expand
#define USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN 1

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
// launch the last three plys in a single kernel (default is to lauch last two plys)
// doesn't really help much in regular positions and even hurts performance in 'good' positions 
//  ~ +4% in start position, -5% in pos2, +20% in pos3, -2.5% in pos4 and pos5
// drastically improves performance (upto 2X) in very bad positions (with very low branching factors)
// with hash tables, could be more helpful in regular positions also
#define PARALLEL_LAUNCH_LAST_3_LEVELS 1
#endif

// flag en-passent capture only when it's possible (in next move)
// default is to flag en-passent on every double pawn push
// This switch works only using using makeMove()
// This helps A LOT when using transposition tables (> 50% improvement in perft(12) time)!
#define EXACT_EN_PASSENT_FLAGGING 1

// combine multiple memory requests into single request to save on number of atomicAdds
// doesn't seem to help at all!
#define COMBINE_DEVICE_MALLOCS 0

// make use of a hash table to avoid duplicate calculations due to transpositions
// it's assumed that INTERVAL_EXPAND is enabled (as it's always used by the hashed perft routine)
#define USE_TRANSPOSITION_TABLE 1

#if USE_TRANSPOSITION_TABLE == 1

// windows 64 bit vs 32 bit vs linux 64 bit compromise :-/
// Windows allows overclocking (gives about 10% extra performance)
// Windows 32 bit build is 7% faster than windows 64 bit build (for some unknown reason??)!
// .. but 32 bit windows build is address space limited.. best settings: don't use sysmem hash, TT_BITS: 25, TT2_BITS: 25, TT3_BITS: 26, TT4_BITS: 25 (total 1.5 GB HASH)
// .. 64 bit windows doesn't allow creating pow-2 vidmem allocations bigger than 1 GB, and sysmem allocations bigger than 2 GB. best settings: TT_BITS: 27, TT2/TT3 BITS: 26, TT4_BITS: 27 (total 4 GB HASH using 2 GB sysmem)
// .. Linux allows using any amount of memory (limited by sysmem/vidmem sizes), so best setting is to distribute all of available vidmem and sysmem to cover all transposition tables adequately

// use system memory hash table (only useful in 64 bit builds otherwise we run of VA space before running out of vid memory)
#define USE_SYSMEM_HASH 1

// A bit risky: use a separate shallow hash table (64-bit entries) for holding depth 5 perfts
// average branching factor of < 55 should be ok using 29 index bits
// ANKAN - this seems buggy, or are we running out of bits? - TODO: debug and fix.
#define USE_SHALLOW_DEPTH5_TT 0

// size of the Deep transposition table (in number of entries)
// maynot be a power of two
// each entry is of 16 bytes
// 28 bits: 256 million entries -> 4 GB
// 27 bits: 128 million entries -> 2 GB
// 25 bits: 32 million entries  -> 512 MB
#define TT_BITS             27
#define TT_SIZE_FROM_BITS   (1ull << TT_BITS)
#define TT_SIZE             (128 * 1024 * 1024)

// bits of the zobrist hash used as index into the transposition table
#define TT_INDEX_BITS  (TT_SIZE_FROM_BITS - 1)

// remaining bits (that are stored per hash entry)
#define TT_HASH_BITS   (ALLSET ^ TT_INDEX_BITS)


// A transposition table for storing positions only at depth 2
// 26 bits: 64 million entries -> 512 MB (each entry is just single uint64: 8 bytes)
// 27 bits: 1 GB
// 29 bits: 4 GB
#define SHALLOW_TT2_BITS         26
#define SHALLOW_TT2_SIZE         (1ull << SHALLOW_TT2_BITS)
#define SHALLOW_TT2_INDEX_BITS   (SHALLOW_TT2_SIZE - 1)
#define SHALLOW_TT2_HASH_BITS    (ALLSET ^ SHALLOW_TT2_INDEX_BITS)



// A transposition table for storing only depth 3 positions
// 26 bits: 64 million entries -> 512 MB (each entry is just single uint64: 8 bytes)
// 27 bits: 1 GB
// 28 bits: 2 GB
// 29 bits: 4 GB
// 30 bits: 8 GB
#define SHALLOW_TT3_BITS         26
#define SHALLOW_TT3_SIZE         (1ull << SHALLOW_TT3_BITS)
#define SHALLOW_TT3_INDEX_BITS   (SHALLOW_TT3_SIZE - 1)
#define SHALLOW_TT3_HASH_BITS    (ALLSET ^ SHALLOW_TT3_INDEX_BITS)

// Transposition table for depth 4
// 28 bits: 2 GB
// 29 bits: 4 GB
#define SHALLOW_TT4_BITS         27
#define SHALLOW_TT4_SIZE         (1ull << SHALLOW_TT4_BITS)
#define SHALLOW_TT4_INDEX_BITS   (SHALLOW_TT4_SIZE - 1)
#define SHALLOW_TT4_HASH_BITS    (ALLSET ^ SHALLOW_TT4_INDEX_BITS)


#if USE_SHALLOW_DEPTH5_TT == 1
// Transposition table for depth 5
// 28 bits: 2 GB
// 29 bits: 4 GB
#define SHALLOW_TT5_BITS         24
#define SHALLOW_TT5_SIZE         (1ull << SHALLOW_TT5_BITS)
#define SHALLOW_TT5_INDEX_BITS   (SHALLOW_TT5_SIZE - 1)
#define SHALLOW_TT5_HASH_BITS    (ALLSET ^ SHALLOW_TT5_INDEX_BITS)
#endif

// for the three shallow transposition tables above, the perft value is stored in index bits
// as perft 2, perft 3 and perft 4 should always fit even in a 26 bit number

#define TT_Entry HashEntryPerft

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
