// this file contains the various compile time settings/swithes

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif


// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 1

// don't call cudaMalloc/cudaFree from device code, 
// suballocate from a pre-allocated buffer instead
// (most of the routines now won't work when this is OFF - i.e, we now rely on this)
#define USE_PREALLOCATED_MEMORY 1

// 768 MB ... to keep space for transposition tables
// just hope that this would be sufficient :'(

// Keeping 768 MB as preallocated memory size allows us to use 1.5 GBs hash table 
// and allows setting cudaLimitDevRuntimeSyncDepth to a decent depth
#define PREALLOCATED_MEMORY_SIZE (1 * 768 * 1024 * 1024)

// 512 KB ought to be enough for holding the stack for the serial part of the gpu perft
#define GPU_SERIAL_PERFT_STACK_SIZE (512 * 1024)

// use constant memory for accessing lookup tables (except for magic tables as they are huge)
// the default is to use texture cache via __ldg instruction
// (doesn't really affect performance either way. Maybe just a tiny bit slower with fancy magics)
#define USE_CONSTANT_MEMORY_FOR_LUT 0

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

// store two positions (most recent and deepest) in every entry of hash table
// default is to store deepest only
#define USE_DUAL_SLOT_TT 0

// use system memory hash table (only useful in 64 bit builds otherwise we run of VA space)
#define USE_SYSMEM_HASH 0

// size of transposition table (in number of entries)
// must be a power of two
// each entry is of 16 bytes
// 27 bits: 128 million entries -> 4 GB hash table (when dual entry is used), or 2 GB when single entry is used
// 24 Bits: 512 MB (when dual entry is used)
// 25 bits: 1 GB (when dual entry is used)
#define TT_BITS     25
#define TT_SIZE     (1 << TT_BITS)

// bits of the zobrist hash used as index into the transposition table
#define TT_INDEX_BITS  (TT_SIZE - 1)

// remaining bits (that are stored per hash entry)
#define TT_HASH_BITS   (ALLSET ^ TT_INDEX_BITS)


// A transposition table for storing positions only at depth 2
// 26 bits: 64 million entries -> 512 MB (each entry is just single uint64: 8 bytes)
// 27 bits: 1 GB
// 29 bits: 4 GB
#define SHALLOW_TT2_BITS         25
#define SHALLOW_TT2_SIZE         (1 << SHALLOW_TT2_BITS)
#define SHALLOW_TT2_INDEX_BITS   (SHALLOW_TT2_SIZE - 1)
#define SHALLOW_TT2_HASH_BITS    (ALLSET ^ SHALLOW_TT2_INDEX_BITS)



// A transposition table for storing only depth 3 positions
// 26 bits: 64 million entries -> 512 MB (each entry is just single uint64: 8 bytes)
// 27 bits: 1 GB
// 28 bits: 2 GB
// 29 bits: 4 GB
// 30 bits: 8 GB
#define SHALLOW_TT3_BITS         25
#define SHALLOW_TT3_SIZE         (1 << SHALLOW_TT3_BITS)
#define SHALLOW_TT3_INDEX_BITS   (SHALLOW_TT3_SIZE - 1)
#define SHALLOW_TT3_HASH_BITS    (ALLSET ^ SHALLOW_TT3_INDEX_BITS)

// Transposition table for depth 4
// 28 bits: 2 GB
// 29 bits: 4 GB
#define SHALLOW_TT4_BITS         26
#define SHALLOW_TT4_SIZE         (1 << SHALLOW_TT4_BITS)
#define SHALLOW_TT4_INDEX_BITS   (SHALLOW_TT4_SIZE - 1)
#define SHALLOW_TT4_HASH_BITS    (ALLSET ^ SHALLOW_TT4_INDEX_BITS)


// for the three shallow transposition tables above, the perft value is stored in index bits
// as perft 2, perft 3 and perft 4 should always fit even in a 26 bit number

#if USE_DUAL_SLOT_TT == 1
#define TT_Entry DualHashEntry
#else
#define TT_Entry HashEntryPerft
#endif

#endif

// count the no of times countMoves() got called (useful to find hash table effectiveness)
#define COUNT_NUM_COUNT_MOVES 0

// print  various hash statistics
#define PRINT_HASH_STATS 0

// move generation functions templated on chance
#define USE_TEMPLATE_CHANCE_OPT 1

// bitwise magic instead of if/else for castle flag updation
#define USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION 1

// intel core 2 doesn't have popcnt instruction
#define USE_POPCNT 0

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
#define USE_SLIDING_LUT 1

// use fancy fixed-shift version - ~ 800 KB lookup tables
// (setting this to 0 enables plain magics - with 2.3 MB lookup table)
// plain magics is a bit faster at least for perft (on core 2 duo)
// fancy magics is clearly faster on more recent processors (ivy bridge)
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
