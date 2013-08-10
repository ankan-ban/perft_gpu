// this file contains the various compile time settings/swithes

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif

// make use of a hash table to avoid duplicate calculations due to transpositions
#define USE_TRANSPOSITION_TABLE 1

// avoid creating huge tables in CPU memory when we aren't using them!
#define USE_TRANSPOSITION_TABLE_FOR_CPU_PERFT 0

#if USE_TRANSPOSITION_TABLE == 1

// incrementally calculate zobrist hash when making a move / generating a board
// currently only works with move list (when USE_MOVE_LIST == 1)
// costs ~7% for CPU perft (on CPU)
#define INCREMENTAL_ZOBRIST_UPDATE 0

// check if the incremental update of zobrist hash is working as expected
#define DEBUG_INCREMENTAL_ZOBRIST_UPDATE 0

// store two positions (most recent and deepest) in every entry of hash table
#define USE_DUAL_SLOT_TT 1

// make use of transposition table even at the leaves
#define USE_TRANSPOSITION_AT_LEAVES 0

// size of transposition table (in number of entries)
// must be a power of two
// each entry is of 16 bytes
// 27 bits: 128 million entries -> 4 GB hash table (when dual entry is used), or 2 GB when single entry is used
// 24 Bits: 512 MB (when dual entry is used)
#define TT_BITS     24
#define TT_SIZE     (1 << TT_BITS)

// bits of the zobrist hash used as index into the transposition table
#define TT_INDEX_BITS  (TT_SIZE - 1)

// remaining bits (that are stored per hash entry)
#define TT_HASH_BITS   (ALLSET ^ TT_INDEX_BITS)

// use a second transposition table for storing positions only at depth 2
#define USE_SHALLOW_TT 1

#if USE_SHALLOW_TT == 1
// 27 bits: 128 million entries -> 1 GB (each entry is just single uint64: 8 bytes)
#define SHALLOW_TT_BITS         27  
#define SHALLOW_TT_SIZE         (1 << SHALLOW_TT_BITS)
#define SHALLOW_TT_INDEX_BITS   (SHALLOW_TT_SIZE - 1)
#define SHALLOW_TT_HASH_BITS    (ALLSET ^ SHALLOW_TT_INDEX_BITS)
#endif

#if USE_DUAL_SLOT_TT == 1
#define TT_Entry DualHashEntry
#else
#define TT_Entry HashEntryPerft
#endif

#endif

// don't call cudaMalloc/cudaFree from device code, 
// suballocate from a pre-allocated buffer instead
#define USE_PREALLOCATED_MEMORY 1

// 384 MB ... to keep space for transposition tables
// just hope that this would be sufficient :'(

// Keeping 384 MB as preallocated memory size allows us to use 2 GB hash table 
// and allows setting cudaLimitDevRuntimeSyncDepth to 5 - which allows 
// parallel kernel launch depth of 7 (when 3 levels opt is enabled) or 6 (when it isn't)
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
#define PARALLEL_LAUNCH_LAST_3_LEVELS 0
#endif

// first add moves to a move list and then use makeMove function to update the board
// when this is set to 0, generateBoards is called to generate the updated boards directly
// Note that this flag is only for CPU perft. For gpu, we always make use of moveList
#define USE_MOVE_LIST_FOR_CPU_PERFT 0

// only count moves at leaves (instead of generating/making them)
#define USE_COUNT_ONLY_OPT 1

// move generation functions templated on chance
#define USE_TEMPLATE_CHANCE_OPT 1

// bitwise magic instead of if/else for castle flag updation
#define USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION 1

// Move counting (countOnly) doesn't work with old method
#define EN_PASSENT_GENERATION_NEW_METHOD 1

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
