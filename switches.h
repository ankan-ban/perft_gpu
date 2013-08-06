// this file contains the various compile time settings/swithes

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif

// don't call cudaMalloc/cudaFree from device code, 
// suballocate from a pre-allocated buffer instead
#define USE_PREALLOCATED_MEMORY 1

// 1 GB for now
#define PREALLOCATED_MEMORY_SIZE (1 * 1024 * 1024 * 1024)

// use constant memory for accessing lookup tables (except for magic tables as they are huge)
// the default is to use texture cache via __ldg instruction
// (doesn't really affect performance either way. Maybe just a tiny bit slower with fancy magics)
#define USE_CONSTANT_MEMORY_FOR_LUT 0

// use parallel scan and interval expand algorithms (from modern gpu lib) for 
// performing the move list scan and 'expand' operation to set correct board pointers for second level child moves
#define USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN 1

#if USE_INTERVAL_EXPAND_FOR_MOVELIST_SCAN == 1
// launch the last three plys in a single kernel (default is to lauch last two plys)
// doesn't really help much in regular positions and even hurts performance in 'good' positions 
//  ~ +4% in start position, -5% in pos2, +20% in pos3, -2.5% in pos4 and pos5
// drastically improves performance (upto 2X) in very bad positions (with very low branching factors)
// with hash tables, could be more helpful in regular positions also
#define PARALLEL_LAUNCH_LAST_3_LEVELS 1
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
