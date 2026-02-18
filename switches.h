// this file contains the various compile time settings/switches

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif


// can be tuned as per need
// 256 works best for Maxwell
// 384 best for newer chips!
// (also make sure max registers used is set to 47 on maxwell and 64 on newer chips) or set max registers to 0 and enable LIMIT_REGISTER_USAGE
#define BLOCK_SIZE 384

// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 1

// 3 works best with 384 block size on new chips
#define  MIN_BLOCKS_PER_MP 3

// preallocated memory size (for holding the perft tree in GPU memory)
#define PREALLOCATED_MEMORY_SIZE (16 * 1024 * 1024 * 1024ull)

// 512 KB ought to be enough for holding the stack for the serial part of the gpu perft
#define GPU_SERIAL_PERFT_STACK_SIZE (512 * 1024)

// use constant memory for accessing lookup tables (except for magic tables as they are huge)
// the default is to use texture cache via __ldg instruction
// Ankan - improves performance on Maxwell a LOT!
//  - but hurts performance on Newer hardware
#define USE_CONSTANT_MEMORY_FOR_LUT 0

// no transposition table in this simplified build
#define USE_TRANSPOSITION_TABLE 0

// move generation functions templated on chance
// +9% benefit on GM204
#define USE_TEMPLATE_CHANCE_OPT 1

// bitwise magic instead of if/else for castle flag updation
// turning this off helps Maxwell a little
#define USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION 0

// intel core 2 doesn't have popcnt instruction
#define USE_POPCNT 0

// pentium 4 doesn't have fast HW bitscan
#define USE_HW_BITSCAN 1

// use lookup tables for figuring out squares in line and squares in between
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
