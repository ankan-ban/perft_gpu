#include "chess.h"
#include "FancyMagics.h"
#include <intrin.h>
#include <time.h>

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
           void   *preAllocatedBufferHost;
__device__ void   *preAllocatedBuffer;
__device__ uint32  preAllocatedMemoryUsed;


// use __shfl to improve memory colaesing when writing board pointers
// improves performance by only 1% :-/
#define USE_COLAESED_WRITES_FOR_MOVELIST_SCAN 1

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
#define USE_SLIDING_LUT 0

// use fancy fixed-shift version - ~ 800 KB lookup tables
// (setting this to 0 enables plain magics - with 2.3 MB lookup table)
// plain magics is a bit faster at least for perft (on core 2 duo)
// fancy magics is clearly faster on more recent processors (ivy bridge)
#define USE_FANCY_MAGICS 1

// bit board constants
#define C64(constantU64) constantU64##ULL

// valid locations for pawns
#define RANKS2TO7 C64(0x00FFFFFFFFFFFF00)

#define RANK1     C64(0x00000000000000FF)
#define RANK2     C64(0x000000000000FF00)
#define RANK3     C64(0x0000000000FF0000)
#define RANK4     C64(0x00000000FF000000)
#define RANK5     C64(0x000000FF00000000)
#define RANK6     C64(0x0000FF0000000000)
#define RANK7     C64(0x00FF000000000000)
#define RANK8     C64(0xFF00000000000000)

#define FILEA     C64(0x0101010101010101)
#define FILEB     C64(0x0202020202020202)
#define FILEC     C64(0x0404040404040404)
#define FILED     C64(0x0808080808080808)
#define FILEE     C64(0x1010101010101010)
#define FILEF     C64(0x2020202020202020)
#define FILEG     C64(0x4040404040404040)
#define FILEH     C64(0x8080808080808080)

#define DIAGONAL_A1H8   C64(0x8040201008040201)
#define DIAGONAL_A8H1   C64(0x0102040810204080)

#define CENTRAL_SQUARES C64(0x007E7E7E7E7E7E00)

// used for castling checks
#define F1G1      C64(0x60)
#define C1D1      C64(0x0C)
#define B1D1      C64(0x0E)

// used for castling checks
#define F8G8      C64(0x6000000000000000)
#define C8D8      C64(0x0C00000000000000)
#define B8D8      C64(0x0E00000000000000)

// used to update castle flags
#define WHITE_KING_SIDE_ROOK   C64(0x0000000000000080)
#define WHITE_QUEEN_SIDE_ROOK  C64(0x0000000000000001)
#define BLACK_KING_SIDE_ROOK   C64(0x8000000000000000)
#define BLACK_QUEEN_SIDE_ROOK  C64(0x0100000000000000)
    

#define ALLSET    C64(0xFFFFFFFFFFFFFFFF)
#define EMPTY     C64(0x0)

CUDA_CALLABLE_MEMBER __forceinline uint8 popCount(uint64 x)
{
#ifdef __CUDA_ARCH__
    return __popcll(x);
#elif USE_POPCNT == 1
#ifdef _WIN64
    return _mm_popcnt_u64(x);
#else
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    return _mm_popcnt_u32(lo) + _mm_popcnt_u32(hi);
#endif
#else

    // taken from chess prgramming wiki: http://chessprogramming.wikispaces.com/Population+Count
    const uint64 k1 = C64(0x5555555555555555); /*  -1/3   */
    const uint64 k2 = C64(0x3333333333333333); /*  -1/5   */
    const uint64 k4 = C64(0x0f0f0f0f0f0f0f0f); /*  -1/17  */
    const uint64 kf = C64(0x0101010101010101); /*  -1/255 */

    x =  x       - ((x >> 1)  & k1); /* put count of each 2 bits into those 2 bits */
    x = (x & k2) + ((x >> 2)  & k2); /* put count of each 4 bits into those 4 bits */
    x = (x       +  (x >> 4)) & k4 ; /* put count of each 8 bits into those 8 bits */
    x = (x * kf) >> 56;              /* returns 8 most significant bits of x + (x<<8) + (x<<16) + (x<<24) + ...  */

    return (uint8) x;
#endif
}


// return the index of first set LSB
CUDA_CALLABLE_MEMBER __forceinline uint8 bitScan(uint64 x)
{
#ifdef __CUDA_ARCH__
    // __ffsll(x) returns position from 1 to 64 instead of 0 to 63
    return __ffsll(x) - 1;
#elif _WIN64
   unsigned long index;
   assert (x != 0);
   _BitScanForward64(&index, x);
   return (uint8) index;    
#else
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    DWORD id; 

    if (lo)
        _BitScanForward(&id, lo);
    else
    {
        _BitScanForward(&id, hi);
        id += 32;
    }

    return (uint8) id; 
#endif
}

// bit mask containing squares between two given squares
static uint64 Between[64][64];

// bit mask containing squares in the same 'line' as two given squares
static uint64 Line[64][64];

// squares a piece can attack in an empty board
static uint64 RookAttacks    [64];
static uint64 BishopAttacks  [64];
static uint64 QueenAttacks   [64];
static uint64 KingAttacks    [64];
static uint64 KnightAttacks  [64];
static uint64 pawnAttacks[2] [64];

// magic lookup tables
// plain magics (Fancy magic lookup tables in FancyMagics.h)
#define ROOK_MAGIC_BITS    12
#define BISHOP_MAGIC_BITS  9

uint64 rookMagics            [64];
uint64 bishopMagics          [64];

// same as RookAttacks and BishopAttacks, but corner bits masked off
static uint64 RookAttacksMasked   [64];
static uint64 BishopAttacksMasked [64];

uint64 rookMagicAttackTables      [64][1 << ROOK_MAGIC_BITS  ];    // 2 MB
uint64 bishopMagicAttackTables    [64][1 << BISHOP_MAGIC_BITS];    // 256 KB

uint64 findRookMagicForSquare(int square, uint64 magicAttackTable[], uint64 magic = 0);
uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 magic = 0);


// gpu version of the above data structures
// accessed for read only using __ldg() function

// bit mask containing squares between two given squares
__device__ static uint64 gBetween[64][64];

// bit mask containing squares in the same 'line' as two given squares
__device__ static uint64 gLine[64][64];

// squares a piece can attack in an empty board
__device__ static uint64 gRookAttacks    [64];
__device__ static uint64 gBishopAttacks  [64];
__device__ static uint64 gQueenAttacks   [64];
__device__ static uint64 gKingAttacks    [64];
__device__ static uint64 gKnightAttacks  [64];
__device__ static uint64 gpawnAttacks[2] [64];


// Magical Tables
// same as RookAttacks and BishopAttacks, but corner bits masked off
__device__ static uint64 gRookAttacksMasked   [64];
__device__ static uint64 gBishopAttacksMasked [64];

// plain magics
__device__ static uint64 gRookMagics                [64];
__device__ static uint64 gBishopMagics              [64];
__device__ static uint64 gRookMagicAttackTables     [64][1 << ROOK_MAGIC_BITS  ];    // 2 MB
__device__ static uint64 gBishopMagicAttackTables   [64][1 << BISHOP_MAGIC_BITS];    // 256 KB

// fancy magics (cpu versions in FancyMagics.h)
__device__ static uint64 g_fancy_magic_lookup_table[97264];
__device__ static FancyMagicEntry g_bishop_magics_fancy[64];
__device__ static FancyMagicEntry g_rook_magics_fancy[64];


CUDA_CALLABLE_MEMBER __forceinline uint64 sqsInBetweenLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBetween[sq1][sq2]);
#else
    return Between[sq1][sq2];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqsInLineLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gLine[sq1][sq2]);
#else
    return Line[sq1][sq2];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqKnightAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gKnightAttacks[sq]);
#else
    return KnightAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqKingAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gKingAttacks[sq]);
#else
    return KingAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqRookAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookAttacks[sq]);
#else
    return RookAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqBishopAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopAttacks[sq]);
#else
    return BishopAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqBishopAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopAttacksMasked[sq]);
#else
    return BishopAttacksMasked[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqRookAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookAttacksMasked[sq]);
#else
    return RookAttacksMasked[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqRookMagics(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookMagics[sq]);
#else
    return rookMagics[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqBishopMagics(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopMagics[sq]);
#else
    return bishopMagics[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqRookMagicAttackTables(uint8 sq, int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookMagicAttackTables[sq][index]);
#else
    return rookMagicAttackTables[sq][index];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sqBishopMagicAttackTables(uint8 sq, int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopMagicAttackTables[sq][index]);
#else
    return bishopMagicAttackTables[sq][index];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline uint64 sq_fancy_magic_lookup_table(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_magic_lookup_table[index]);
#else
    return fancy_magic_lookup_table[index];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline FancyMagicEntry sq_bishop_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
    FancyMagicEntry op;
    op.factor   = __ldg(&g_bishop_magics_fancy[sq].factor);
    op.position = __ldg(&g_bishop_magics_fancy[sq].position);
    return op;
#else
    return bishop_magics_fancy[sq];
#endif
}

CUDA_CALLABLE_MEMBER __forceinline FancyMagicEntry sq_rook_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
    FancyMagicEntry op;
    op.factor   = __ldg(&g_rook_magics_fancy[sq].factor);
    op.position = __ldg(&g_rook_magics_fancy[sq].position);
    return op;
#else
    return rook_magics_fancy[sq];
#endif
}

class MoveGeneratorBitboard
{
public:

    // move the bits in the bitboard one square in the required direction

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northOne(uint64 x)
    {
        return x << 8;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southOne(uint64 x)
    {
        return x >> 8;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 eastOne(uint64 x)
    {
        return (x << 1) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 westOne(uint64 x)
    {
        return (x >> 1) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northEastOne(uint64 x)
    {
        return (x << 9) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northWestOne(uint64 x)
    {
        return (x << 7) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southEastOne(uint64 x)
    {
        return (x >> 7) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southWestOne(uint64 x)
    {
        return (x >> 9) & (~FILEH);
    }


    // fill the board in the given direction
    // taken from http://chessprogramming.wikispaces.com/


    // gen - generator  : starting positions
    // pro - propogator : empty squares / squares not of current side

    // uses kogge-stone algorithm

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northFill(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southFill(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 eastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return gen;
    }
    
    CUDA_CALLABLE_MEMBER __forceinline static uint64 westFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return gen;
    }


    CUDA_CALLABLE_MEMBER __forceinline static uint64 northEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northWestFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southWestFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 9) & pro;
        pro &= (pro >> 9);
        gen |= (gen >> 18) & pro;
        pro &= (pro >> 18);
        gen |= (gen >> 36) & pro;

        return gen;
    }


    // attacks in the given direction
    // need to OR with ~(pieces of side to move) to avoid killing own pieces

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen << 8;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen >> 8;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 eastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return (gen << 1) & (~FILEA);
    }
    
    CUDA_CALLABLE_MEMBER __forceinline static uint64 westAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return (gen >> 1) & (~FILEH);
    }


    CUDA_CALLABLE_MEMBER __forceinline static uint64 northEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return (gen << 9) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 northWestAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return (gen << 7) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return (gen >> 7) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 southWestAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 9) & pro;
        pro &= (pro >> 9);
        gen |= (gen >> 18) & pro;
        pro &= (pro >> 18);
        gen |= (gen >> 36) & pro;

        return (gen >> 9) & (~FILEH);
    }


    // attacks by pieces of given type
    // pro - empty squares

    CUDA_CALLABLE_MEMBER __forceinline static uint64 bishopAttacksKoggeStone(uint64 bishops, uint64 pro)
    {
        return northEastAttacks(bishops, pro) |
               northWestAttacks(bishops, pro) |
               southEastAttacks(bishops, pro) |
               southWestAttacks(bishops, pro) ;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 rookAttacksKoggeStone(uint64 rooks, uint64 pro)
    {
        return northAttacks(rooks, pro) |
               southAttacks(rooks, pro) |
               eastAttacks (rooks, pro) |
               westAttacks (rooks, pro) ;
    }


#if USE_SLIDING_LUT == 1
    CUDA_CALLABLE_MEMBER __forceinline static uint64 bishopAttacks(uint64 bishop, uint64 pro)
    {
        uint8 square = bitScan(bishop);
        uint64 occ = (~pro) & sqBishopAttacksMasked(square);

#if USE_FANCY_MAGICS == 1
        FancyMagicEntry magicEntry = sq_bishop_magics_fancy(square);
        uint64 index = (magicEntry.factor * occ) >> (64 - BISHOP_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#else
        uint64 magic = sqBishopMagics(square);
        uint64 index = (magic * occ) >> (64 - BISHOP_MAGIC_BITS);
        return sqBishopMagicAttackTables(square, index);
#endif
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 rookAttacks(uint64 rook, uint64 pro)
    {
        uint8 square = bitScan(rook);
        uint64 occ = (~pro) & sqRookAttacksMasked(square);

#if USE_FANCY_MAGICS == 1
        FancyMagicEntry magicEntry = sq_rook_magics_fancy(square);
        uint64 index = (magicEntry.factor * occ) >> (64 - ROOK_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#else
        uint64 magic = sqRookMagics(square);
        uint64 index = (magic * occ) >> (64 - ROOK_MAGIC_BITS);
        return sqRookMagicAttackTables(square, index);
#endif
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 multiBishopAttacks(uint64 bishops, uint64 pro)
    {
        uint64 attacks = 0;
        while(bishops)
        {
            uint64 bishop = getOne(bishops);
            attacks |= bishopAttacks(bishop, pro);
            bishops ^= bishop;
        }

        return attacks;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 multiRookAttacks(uint64 rooks, uint64 pro)
    {
        uint64 attacks = 0;
        while(rooks)
        {
            uint64 rook = getOne(rooks);
            attacks |= rookAttacks(rook, pro);
            rooks ^= rook;
        }

        return attacks;
    }
#else
// kogge stone handles multiple attackers automatically

#define bishopAttacks bishopAttacksKoggeStone
#define rookAttacks   rookAttacksKoggeStone

#define multiBishopAttacks bishopAttacksKoggeStone
#define multiRookAttacks   rookAttacksKoggeStone
#endif

// not used
#if 0
    CUDA_CALLABLE_MEMBER __forceinline static uint64 queenAttacks(uint64 queens, uint64 pro)
    {
        return rookAttacks  (queens, pro) |
               bishopAttacks(queens, pro) ;
    }
#endif

    CUDA_CALLABLE_MEMBER __forceinline static uint64 kingAttacks(uint64 kingSet) 
    {
        uint64 attacks = eastOne(kingSet) | westOne(kingSet);
        kingSet       |= attacks;
        attacks       |= northOne(kingSet) | southOne(kingSet);
        return attacks;
    }

    // efficient knight attack generator
    // http://chessprogramming.wikispaces.com/Knight+Pattern
    CUDA_CALLABLE_MEMBER __forceinline static uint64 knightAttacks(uint64 knights) {
        uint64 l1 = (knights >> 1) & C64(0x7f7f7f7f7f7f7f7f);
        uint64 l2 = (knights >> 2) & C64(0x3f3f3f3f3f3f3f3f);
        uint64 r1 = (knights << 1) & C64(0xfefefefefefefefe);
        uint64 r2 = (knights << 2) & C64(0xfcfcfcfcfcfcfcfc);
        uint64 h1 = l1 | r1;
        uint64 h2 = l2 | r2;
        return (h1<<16) | (h1>>16) | (h2<<8) | (h2>>8);
    }



    // gets one bit (the LSB) from a bitboard
    // returns a bitboard containing that bit
    CUDA_CALLABLE_MEMBER __forceinline static uint64 getOne(uint64 x)
    {
        return x & (-x);
    }

    CUDA_CALLABLE_MEMBER __forceinline static bool isMultiple(uint64 x)
    {
        return x ^ getOne(x);
    }

    CUDA_CALLABLE_MEMBER __forceinline static bool isSingular(uint64 x)
    {
        return !isMultiple(x); 
    }


    // finds the squares in between the two given squares
    // taken from 
    // http://chessprogramming.wikispaces.com/Square+Attacked+By#Legality Test-In Between-Pure Calculation
    // Ankan : TODO: this doesn't seem to work for G8 - B3
    CUDA_CALLABLE_MEMBER __forceinline static uint64 squaresInBetween(uint8 sq1, uint8 sq2)
    {
        const uint64 m1   = C64(0xFFFFFFFFFFFFFFFF);
        const uint64 a2a7 = C64(0x0001010101010100);
        const uint64 b2g7 = C64(0x0040201008040200);
        const uint64 h1b7 = C64(0x0002040810204080);
        uint64 btwn, line, rank, file;
     
        btwn  = (m1 << sq1) ^ (m1 << sq2);
        file  =   (sq2 & 7) - (sq1   & 7);
        rank  =  ((sq2 | 7) -  sq1) >> 3 ;
        line  =      (   (file  &  7) - 1) & a2a7; // a2a7 if same file
        line += 2 * ((   (rank  &  7) - 1) >> 58); // b1g1 if same rank
        line += (((rank - file) & 15) - 1) & b2g7; // b2g7 if same diagonal
        line += (((rank + file) & 15) - 1) & h1b7; // h1b7 if same antidiag
        line *= btwn & -btwn; // mul acts like shift by smaller square
        return line & btwn;   // return the bits on that line inbetween
    }

    // returns the 'line' containing all pieces in the same file/rank/diagonal or anti-diagonal containing sq1 and sq2
    CUDA_CALLABLE_MEMBER __forceinline static uint64 squaresInLine(uint8 sq1, uint8 sq2)
    {
        // TODO: try to make it branchless?
        int fileDiff  =   (sq2 & 7) - (sq1 & 7);
        int rankDiff  =  ((sq2 | 7) -  sq1) >> 3 ;

        uint8 file = sq1 & 7;
        uint8 rank = sq1 >> 3;

        if (fileDiff == 0)  // same file
        {
            return FILEA << file;
        }
        if (rankDiff == 0)  // same rank
        {
            return RANK1 << (rank * 8);
        }
        if (fileDiff - rankDiff == 0)   // same diagonal (with slope equal to a1h8)
        {
            if (rank - file >= 0)
                return DIAGONAL_A1H8 << ((rank - file) * 8);
            else
                return DIAGONAL_A1H8 >> ((file - rank) * 8);
        }
        if (fileDiff + rankDiff == 0)  // same anti-diagonal (with slope equal to a8h1)
        {
            // for a8h1, rank + file = 7
            int shiftAmount = (rank + file - 7) * 8;
            if (shiftAmount >= 0)
                return DIAGONAL_A8H1 << shiftAmount;
            else
                return DIAGONAL_A8H1 >> (-shiftAmount);
        }

        // squares not on same line
        return 0;
    }


    CUDA_CALLABLE_MEMBER __forceinline static uint64 sqsInBetween(uint8 sq1, uint8 sq2)
    {
#if USE_IN_BETWEEN_LUT == 1
        return sqsInBetweenLUT(sq1, sq2);
#else
        return squaresInBetween(sq1, sq2);
#endif
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 sqsInLine(uint8 sq1, uint8 sq2)
    {
#if USE_IN_BETWEEN_LUT == 1
        return sqsInLineLUT(sq1, sq2);
#else
        return squaresInLine(sq1, sq2);
#endif
    }


    static void init()
    {
        // initialize the empty board attack tables
        for (uint8 i=0; i < 64; i++)
        {
            uint64 x = BIT(i);
            uint64 north = northAttacks(x, ALLSET);
            uint64 south = southAttacks(x, ALLSET);
            uint64 east  = eastAttacks (x, ALLSET);
            uint64 west  = westAttacks (x, ALLSET);
            uint64 ne    = northEastAttacks(x, ALLSET);
            uint64 nw    = northWestAttacks(x, ALLSET);
            uint64 se    = southEastAttacks(x, ALLSET);
            uint64 sw    = southWestAttacks(x, ALLSET);
            
            RookAttacks  [i] = north | south | east | west;
            BishopAttacks[i] = ne | nw | se | sw;
            QueenAttacks [i] = RookAttacks[i] | BishopAttacks[i];
            KnightAttacks[i] = knightAttacks(x);
            KingAttacks[i]   = kingAttacks(x);

            // TODO: initialize pawn attack table
            // probably not really needed (as pawn attack calculation is simple enough)
        }

        // initialize the Between and Line tables
        for (uint8 i=0; i<64; i++)
            for (uint8 j=0; j<64; j++)
            {
                if (i <= j)
                {
                    Between[i][j] = squaresInBetween(i, j);
                    Between[j][i] = Between[i][j];
                }
                Line[i][j] = squaresInLine(i, j);
            }


        // initialize magic lookup tables
#if USE_SLIDING_LUT == 1
        srand (time(NULL));
        for (int square = A1; square <= H8; square++)
        {
            uint64 thisSquare = BIT(square);
            uint64 mask    = sqRookAttacks(square) & (~thisSquare);

            // mask off squares that don't matter
            if ((thisSquare & RANK1) == 0)
                mask &= ~RANK1;

            if ((thisSquare & RANK8) == 0)
                mask &= ~RANK8;

            if ((thisSquare & FILEA) == 0)
                mask &= ~FILEA;

            if ((thisSquare & FILEH) == 0)
                mask &= ~FILEH;
            
            RookAttacksMasked[square] = mask;

            mask = sqBishopAttacks(square)  & (~thisSquare) & CENTRAL_SQUARES;
            BishopAttacksMasked[square] = mask;
#if USE_FANCY_MAGICS != 1
            rookMagics  [square] = findRookMagicForSquare  (square, rookMagicAttackTables  [square]);
            bishopMagics[square] = findBishopMagicForSquare(square, bishopMagicAttackTables[square]);
#endif
        }

        // initialize fancy magic lookup table
        for (int square = A1; square <= H8; square++)
        {
            uint64 rookMagic = findRookMagicForSquare  (square, &fancy_magic_lookup_table[rook_magics_fancy[square].position], rook_magics_fancy[square].factor);
            assert(rookMagic == rook_magics_fancy[square].factor);

            uint64 bishopMagic = findBishopMagicForSquare  (square, &fancy_magic_lookup_table[bishop_magics_fancy[square].position], bishop_magics_fancy[square].factor);
            assert(bishopMagic == bishop_magics_fancy[square].factor);
        }
#endif        

        // copy all the lookup tables from CPU's memory to GPU memory
        cudaError_t err = cudaMemcpyToSymbol(gBetween, Between, sizeof(Between));
        if (err != S_OK) printf("For copying between table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

        err = cudaMemcpyToSymbol(gLine, Line, sizeof(Line));
        if (err != S_OK) printf("For copying line table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gRookAttacks, RookAttacks, sizeof(RookAttacks));
        if (err != S_OK) printf("For copying RookAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gBishopAttacks, BishopAttacks, sizeof(BishopAttacks));
        if (err != S_OK) printf("For copying BishopAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  
        
        err = cudaMemcpyToSymbol(gQueenAttacks, QueenAttacks, sizeof(QueenAttacks));
        if (err != S_OK) printf("For copying QueenAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gKnightAttacks, KnightAttacks, sizeof(KnightAttacks));
        if (err != S_OK) printf("For copying KnightAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gKingAttacks, KingAttacks, sizeof(KingAttacks));
        if (err != S_OK) printf("For copying KingAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        // Copy magical tables
        err = cudaMemcpyToSymbol(gRookAttacksMasked, RookAttacksMasked, sizeof(RookAttacksMasked));
        if (err != S_OK) printf("For copying RookAttacksMasked table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gBishopAttacksMasked, BishopAttacksMasked , sizeof(BishopAttacksMasked));
        if (err != S_OK) printf("For copying BishopAttacksMasked  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  
        
        err = cudaMemcpyToSymbol(gRookMagics, rookMagics, sizeof(rookMagics));
        if (err != S_OK) printf("For copying rookMagics  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gBishopMagics, bishopMagics, sizeof(bishopMagics));
        if (err != S_OK) printf("For copying bishopMagics table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gRookMagicAttackTables, rookMagicAttackTables, sizeof(rookMagicAttackTables));
        if (err != S_OK) printf("For copying RookMagicAttackTables, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gBishopMagicAttackTables, bishopMagicAttackTables, sizeof(bishopMagicAttackTables));
        if (err != S_OK) printf("For copying bishopMagicAttackTables, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(g_fancy_magic_lookup_table, fancy_magic_lookup_table, sizeof(fancy_magic_lookup_table));
        if (err != S_OK) printf("For copying fancy_magic_lookup_table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(g_bishop_magics_fancy, bishop_magics_fancy, sizeof(bishop_magics_fancy));
        if (err != S_OK) printf("For copying bishop_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(g_rook_magics_fancy, rook_magics_fancy, sizeof(rook_magics_fancy));
        if (err != S_OK) printf("For copying rook_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

    }


    CUDA_CALLABLE_MEMBER static __forceinline uint64 findPinnedPieces (uint64 myKing, uint64 myPieces, uint64 enemyBishops, uint64 enemyRooks, uint64 allPieces, uint8 kingIndex)
    {
        // check for sliding attacks to the king's square

        // It doesn't matter if we process more attackers behind the first attackers
        // They will be taken care of when we check for no. of obstructing squares between king and the attacker
        /*
        uint64 b = bishopAttacks(myKing, ~enemyPieces) & enemyBishops;
        uint64 r = rookAttacks  (myKing, ~enemyPieces) & enemyRooks;
        */

        uint64 b = sqBishopAttacks(kingIndex) & enemyBishops;
        uint64 r = sqRookAttacks  (kingIndex) & enemyRooks;

        uint64 attackers = b | r;

        // for every attacker we need to chess if there is a single obstruction between 
        // the attacker and the king, and if so - the obstructor is pinned
        uint64 pinned = EMPTY;
        while (attackers)
        {
            uint64 attacker = getOne(attackers);

            // bitscan shouldn't be too expensive but it will be good to 
            // figure out a way do find obstructions without having to get square index of attacker
            uint8 attackerIndex = bitScan(attacker);    // same as bitscan on attackers

            uint64 squaresInBetween = sqsInBetween(attackerIndex, kingIndex); // same as using obstructed() function
            uint64 piecesInBetween = squaresInBetween & allPieces;
            if (isSingular(piecesInBetween))
                pinned |= piecesInBetween;

            attackers ^= attacker;  // same as &= ~attacker
        }

        return pinned;
    }

    // returns bitmask of squares in threat by enemy pieces
    // the king shouldn't ever attempt to move to a threatened square
    // TODO: maybe make this tempelated on color?
    CUDA_CALLABLE_MEMBER __forceinline static uint64 findAttackedSquares(uint64 emptySquares, uint64 enemyBishops, uint64 enemyRooks, 
                                      uint64 enemyPawns, uint64 enemyKnights, uint64 enemyKing, 
                                      uint64 myKing, uint8 enemyColor)
    {
        uint64 attacked = 0;

        // 1. pawn attacks
        if (enemyColor == WHITE)
        {
            attacked |= northEastOne(enemyPawns);
            attacked |= northWestOne(enemyPawns);
        }
        else
        {
            attacked |= southEastOne(enemyPawns);
            attacked |= southWestOne(enemyPawns);
        }

        // 2. knight attacks
        attacked |= knightAttacks(enemyKnights);
        
        // 3. bishop attacks
        attacked |= multiBishopAttacks(enemyBishops, emptySquares | myKing); // squares behind king are also under threat (in the sense that king can't go there)

        // 4. rook attacks
        attacked |= multiRookAttacks(enemyRooks, emptySquares | myKing); // squares behind king are also under threat

        // 5. King attacks
        attacked |= kingAttacks(enemyKing);
        
        // TODO: 
        // 1. figure out if we really need to mask off pieces on board
        //  - actually it seems better not to.. so that we can easily check if a capture move takes the king to check
        // 2. It might be faster to use the lookup table instead of computing (esp for king and knights)
        return attacked/*& (emptySquares)*/;
    }


    // adds the given board to list and increments the move counter
    CUDA_CALLABLE_MEMBER __forceinline static void addMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *newBoard)
    {
        **newPos = *newBoard;
        (*newPos)++;
        (*nMoves)++;
    }


    CUDA_CALLABLE_MEMBER __forceinline static void updateCastleFlag(HexaBitBoardPosition *pos, uint64 dst, uint8 chance)
    {

#if USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION == 1
        if (chance == WHITE)
        {
            pos->blackCastle &= ~( ((dst & BLACK_KING_SIDE_ROOK ) >> H8)      |
                                   ((dst & BLACK_QUEEN_SIDE_ROOK) >> (A8-1))) ;
        }
        else
        {
            pos->whiteCastle &= ~( ((dst & WHITE_KING_SIDE_ROOK ) >> H1) |
                                   ((dst & WHITE_QUEEN_SIDE_ROOK) << 1)) ;
        }
#else
        if (chance == WHITE)
        {
            if (dst & BLACK_KING_SIDE_ROOK)
                pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (dst & BLACK_QUEEN_SIDE_ROOK)
                pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
        else
        {
            if (dst & WHITE_KING_SIDE_ROOK)
                pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (dst & WHITE_QUEEN_SIDE_ROOK)
                pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
#endif
    }

    CUDA_CALLABLE_MEMBER __forceinline static void addSlidingMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                             uint64 src, uint64 dst, uint8 chance)
    {

#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->rookQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights) & dst);
            Utils::displayMoveBB(move);
        }
#endif

        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.kings        = pos->kings        & ~dst;
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // figure out if the piece was a bishop, rook, or a queen
        uint64 isBishop = newBoard.bishopQueens & src;
        uint64 isRook   = newBoard.rookQueens   & src;

        // remove src from the appropriate board / boards if queen
        newBoard.bishopQueens ^= isBishop;
        newBoard.rookQueens   ^= isRook;

        // add dst
        newBoard.bishopQueens |= isBishop ? dst : 0;
        newBoard.rookQueens   |= isRook   ? dst : 0;

        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        //newBoard.halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture

        // need to update castle flag for both sides (src moved in same side, and dst move on other side)
        updateCastleFlag(&newBoard, dst,  chance);
        updateCastleFlag(&newBoard, src, !chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    CUDA_CALLABLE_MEMBER __forceinline static void addKnightMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                            uint64 src, uint64 dst, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights | pos->rookQueens) & dst);
            Utils::displayMoveBB(move);
        }
#endif
        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.kings        = pos->kings        & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // remove src and add destination
        newBoard.knights      = (pos->knights ^ src) | dst;

        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        //newBoard.halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture
        updateCastleFlag(&newBoard, dst, chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    CUDA_CALLABLE_MEMBER __forceinline static void addKingMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                          uint64 src, uint64 dst, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights | pos->rookQueens) & dst);
            Utils::displayMoveBB(move);
        }
#endif
        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // remove src and add destination
        newBoard.kings = (pos->kings ^ src) | dst;
        
        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
            newBoard.whiteCastle = 0;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
            newBoard.blackCastle = 0;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        // newBoard.halfMoveCounter++;   // quiet move -> increment half move counter (TODO: fix this for captures)
        updateCastleFlag(&newBoard, dst, chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    CUDA_CALLABLE_MEMBER __forceinline static void addCastleMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                            uint64 kingFrom, uint64 kingTo, uint64 rookFrom, uint64 rookTo, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(kingFrom);
            move.dst = bitScan(kingTo);
            move.flags = 0;
            move.capturedPiece = 0;
            Utils::displayMoveBB(move);
        }
#endif
        HexaBitBoardPosition newBoard;
        newBoard.bishopQueens = pos->bishopQueens;
        newBoard.pawns = pos->pawns;
        newBoard.knights = pos->knights;
        newBoard.kings = (pos->kings ^ kingFrom) | kingTo;
        newBoard.rookQueens = (pos->rookQueens ^ rookFrom) | rookTo;

        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        newBoard.halfMoveCounter = 0;
        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ (kingFrom | rookFrom)) | (kingTo | rookTo);
            newBoard.whiteCastle = 0;
        }
        else
        {
            newBoard.blackCastle = 0;
            newBoard.whitePieces = pos->whitePieces;
        }

        // add the move
        addMove(nMoves, newPos, &newBoard);

    }


    // only for normal moves
    // promotions and en-passent handled in seperate functions
    CUDA_CALLABLE_MEMBER __forceinline static void addSinglePawnMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                               uint64 src, uint64 dst, uint8 chance, bool doublePush, uint8 pawnIndex)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->rookQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights) & dst);;
            Utils::displayMoveBB(move);
        }
#endif
        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.kings        = pos->kings;         // no need to remove dst from kings bitboard as you can't kill a king

        // remove src and add destination
        newBoard.pawns = (pos->pawns ^ src) | dst;
        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
        }

        // no need to update castle flag if the captured piece is a rook
        // as normal pawn moves (except for promotion) can't capture a rook from it's base position

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        if (doublePush)
        {
            newBoard.enPassent = (pawnIndex & 7) + 1;   // store file + 1
        }
        else
        {
            newBoard.enPassent = 0;
        }

        newBoard.halfMoveCounter = 0;   // reset half move counter for pawn push

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }

    CUDA_CALLABLE_MEMBER static void addEnPassentMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                 uint64 src, uint64 dst, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = 1;
            Utils::displayMoveBB(move);
        }
#endif
        HexaBitBoardPosition newBoard;

        uint64 capturedPiece = (chance == WHITE) ? southOne(dst) : northOne(dst);

        newBoard.bishopQueens = pos->bishopQueens;
        newBoard.rookQueens   = pos->rookQueens;
        newBoard.knights      = pos->knights;
        newBoard.kings        = pos->kings;


        // remove src and captured piece. Add destination 
        newBoard.pawns = (pos->pawns ^ (capturedPiece | src)) | dst;
        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  ^ capturedPiece;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.halfMoveCounter = 0;   // reset half move counter for en-passent
        newBoard.enPassent = 0;

        // no need to update castle flag for en-passent

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }

    // adds promotions if at promotion square
    // or normal pawn moves if not promotion. Never called for double pawn push (the above function is called directly)
    CUDA_CALLABLE_MEMBER __forceinline static void addPawnMoves(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                           uint64 src, uint64 dst, uint8 chance)
    {
        // promotion
        if (dst & (RANK1 | RANK8))
        {
#if DEBUG_PRINT_MOVES == 1
            if (printMoves)
            {
                Move move;
                move.src = bitScan(src);
                move.dst = bitScan(dst);
                move.flags = 0;
                move.capturedPiece = 0;
                Utils::displayMoveBB(move);
                Utils::displayMoveBB(move);
                Utils::displayMoveBB(move);
                Utils::displayMoveBB(move);
            }
#endif
            HexaBitBoardPosition newBoard;

            // remove the dst from all bitboards
            newBoard.kings        = pos->kings;         // no need to remove dst from kings bitboard as you can't kill a king

            // remove src and add dst
            if (chance == WHITE)
            {
                newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
            }
            else
            {
                newBoard.whitePieces  = pos->whitePieces  & ~dst;
            }

            // remove src pawn
            newBoard.pawns = (pos->pawns ^ src);

            // update game state (the old game state already got copied over above when copying pawn bitboard)
            newBoard.chance = !chance;
            newBoard.enPassent = 0;
            newBoard.halfMoveCounter = 0;   // reset half move counter for pawn push
            updateCastleFlag(&newBoard, dst, chance);

            // add the moves
            // 1. promotion to knight
            newBoard.knights      = pos->knights      | dst;
            newBoard.bishopQueens = pos->bishopQueens & ~dst;
            newBoard.rookQueens   = pos->rookQueens   & ~dst;
            addMove(nMoves, newPos, &newBoard);

            // 2. promotion to bishop
            newBoard.knights      = pos->knights      & ~dst;
            newBoard.bishopQueens = pos->bishopQueens | dst;
            newBoard.rookQueens   = pos->rookQueens   & ~dst;
            addMove(nMoves, newPos, &newBoard);

            // 3. promotion to queen
            newBoard.rookQueens   = pos->rookQueens   | dst;
            addMove(nMoves, newPos, &newBoard);

            // 4. promotion to rook
            newBoard.bishopQueens = pos->bishopQueens & ~dst;
            addMove(nMoves, newPos, &newBoard);            

        }
        else
        {
            // pawn index is used only for double-pushes (to set en-passent square)
            addSinglePawnMove(nMoves, newPos, pos, src, dst, chance, false, 0);
        }
    }


    CUDA_CALLABLE_MEMBER __forceinline static void addCompactMove(uint32 *nMoves, CMove **genMoves, uint8 from, uint8 to, uint8 flags)
    {
        CMove move(from, to, flags);
        **genMoves = move;
        (*genMoves)++;
        (*nMoves)++;
    }


    // adds promotions if at promotion square
    // or normal pawn moves if not promotion
    CUDA_CALLABLE_MEMBER __forceinline static void addCompactPawnMoves(uint32 *nMoves, CMove **genMoves, uint8 from, uint64 dst, uint8 flags)
    {
        uint8 to = bitScan(dst); 
        // promotion
        if (dst & (RANK1 | RANK8))
        {
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_KNIGHT_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_BISHOP_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_QUEEN_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_ROOK_PROMOTION);
        }
        else
        {
            addCompactMove(nMoves, genMoves, from, to, flags);
        }
    }


#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance>
#endif
    CUDA_CALLABLE_MEMBER __forceinline static uint32 generateBoardsOutOfCheck (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened, 
                                           uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance
#endif
                                           )
    {
        uint32 nMoves = 0;
        uint64 king = pos->kings & myPieces;

        // figure out the no. of attackers 
        uint64 attackers = 0;

        // pawn attacks
        uint64 enemyPawns = allPawns & enemyPieces;
        attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) :
                                          (southEastOne(king) | southWestOne(king)) ) & enemyPawns;

        // knight attackers
        uint64 enemyKnights = pos->knights & enemyPieces;
        attackers |= knightAttacks(king) & enemyKnights;

        // bishop attackers
        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;

        // rook attackers
        uint64 enemyRooks = pos->rookQueens & enemyPieces;
        attackers |= rookAttacks(king, ~allPieces) & enemyRooks;


        // A. Try king moves to get the king out of check
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(king);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while(kingMoves)
        {
            uint64 dst = getOne(kingMoves);
            addKingMove(&nMoves, &newPositions, pos, king, dst, chance);            
            kingMoves ^= dst;
        }


        // B. try moves to kill/block attacking pieces
        if (isSingular(attackers))
        {
            // Find the safe squares - i.e, if a dst square of a move is any of the safe squares, 
            // it will take king out of check

            // for pawn and knight attack, the only option is to kill the attacking piece
            // for bishops rooks and queens, it's the line between the attacker and the king, including the attacker
            uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
            
            // pieces that are pinned don't have any hope of saving the king
            // TODO: Think more about it
            myPieces &= ~pinned;

            // 1. pawn moves
            uint64 myPawns = allPawns & myPieces;

            // checking rank for pawn double pushes
            uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

            uint64 enPassentTarget = 0;
            if (pos->enPassent)
            {
                if (chance == BLACK)
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
                }
                else
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
                }
            }

            // en-passent can only save the king if the piece captured is the attacker
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
            if (enPassentCapturedPiece != attackers)
                enPassentTarget = 0;

            while (myPawns)
            {
                uint64 pawn = getOne(myPawns);

                // pawn push
                uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
                if (dst) 
                {
                    if (dst & safeSquares)
                    {
                        addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
                    }
                    else
                    {
                        // double push (only possible if single push was possible and single push didn't save the king)
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);

                        if (dst) 
                        {
                            addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, bitScan(pawn));
                        }
                    }
                }

                // captures (only one of the two captures will save the king.. if at all it does)
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst) 
                {
                    addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
                }

                // en-passent 
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst) 
                {
                    addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
                }

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (pos->knights & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
                uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & safeSquares;
#else
                uint64 knightMoves = knightAttacks(knight) & safeSquares;
#endif
                while (knightMoves)
                {
                    uint64 dst = getOne(knightMoves);
                    addKnightMove(&nMoves, &newPositions, pos, knight, dst, chance);            
                    knightMoves ^= dst;
                }
                myKnights ^= knight;
            }
            
            // 3. bishop moves
            uint64 bishops = pos->bishopQueens & myPieces;
            while (bishops)
            {
                uint64 bishop = getOne(bishops);
                uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;

                while (bishopMoves)
                {
                    uint64 dst = getOne(bishopMoves);
                    addSlidingMove(&nMoves, &newPositions, pos, bishop, dst, chance);            
                    bishopMoves ^= dst;
                }
                bishops ^= bishop;
            }

            // 4. rook moves
            uint64 rooks = pos->rookQueens & myPieces;
            while (rooks)
            {
                uint64 rook = getOne(rooks);
                uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;

                while (rookMoves)
                {
                    uint64 dst = getOne(rookMoves);
                    addSlidingMove(&nMoves, &newPositions, pos, rook, dst, chance);            
                    rookMoves ^= dst;
                }
                rooks ^= rook;
            }

        }   // end of if single attacker
        else
        {
            // multiple threats => only king moves possible
        }

        return nMoves;
    }


    // generates moves for the given board position
    // returns the no of moves generated
    // newPositions contains the new positions after making the generated moves
    // returns only count if newPositions is NULL
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER static uint32 generateBoards (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions)
#else
    CUDA_CALLABLE_MEMBER static uint32 generateBoards (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions, uint8 chance)
#endif
    {

        uint32 nMoves = 0;

        uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables

        uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
        uint64 blackPieces  = allPieces & (~pos->whitePieces);
        
        uint64 myPieces     = (chance == WHITE) ? pos->whitePieces : blackPieces;
        uint64 enemyPieces  = (chance == WHITE) ? blackPieces      : pos->whitePieces;

        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        uint64 enemyRooks   = pos->rookQueens & enemyPieces;

        uint64 myKing     = pos->kings & myPieces;
        uint8  kingIndex  = bitScan(myKing);

        uint64 pinned     = findPinnedPieces(pos->kings & myPieces, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);

        uint64 threatened = findAttackedSquares(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, 
                                                pos->knights & enemyPieces, pos->kings & enemyPieces, 
                                                myKing, !chance);



        // king is in check: call special generate function to generate only the moves that take king out of check
        if (threatened & (pos->kings & myPieces))
        {
#if USE_TEMPLATE_CHANCE_OPT == 1
            return generateBoardsOutOfCheck<chance>(pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, 
                                                              pinned, threatened, kingIndex);
#else
            return generateBoardsOutOfCheck (pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, 
                                            pinned, threatened, kingIndex, chance);
#endif
        }

        uint64 myPawns = allPawns & myPieces;

        // 0. generate en-passent moves first
        uint64 enPassentTarget = 0;
        if (pos->enPassent)
        {
            if (chance == BLACK)
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            }
            else
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
            }
        }
#if EN_PASSENT_GENERATION_NEW_METHOD == 1
        if (enPassentTarget)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);

            uint64 epSources = ((chance == WHITE) ? southEastOne(enPassentTarget) | southWestOne(enPassentTarget) : 
                                                    northEastOne(enPassentTarget) | northWestOne(enPassentTarget)) & myPawns;

            while (epSources)
            {
                uint64 pawn = getOne(epSources);
                if (pawn & pinned)
                {
                    // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
                    uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                    
                    if (enPassentTarget & line)
                    {
                        addEnPassentMove(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
                    }
                }
                else 
                /*if (!(enPassentCapturedPiece & pinned))*/   
                // the captured pawn should not be pinned in diagonal direction but it can be in vertical dir.
                // the diagonal pinning can't happen for enpassent in real chess game, so anyways it's not vaild
                {
                    uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                    uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                         (pos->kings & myPieces);
                    if (!causesCheck)
                    {
                        addEnPassentMove(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
                    }
                }
                epSources ^= pawn;
            }
        }
#endif
        // 1. pawn moves

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = sqsInLine(pawnIndex, kingIndex);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
            if (dst) 
            {
                addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, false, pawnIndex);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) 
                {
                    addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, pawnIndex);
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
            }

            // en-passent capture isn't possible by a pinned pawn
            // TODO: think more about it
            // it's actually possible, if the pawn moves in the 'direction' of the pin
            // check out the position: rnb1kb1r/ppqp1ppp/2p5/4P3/2B5/6K1/PPP1N1PP/RNBQ3R b kq - 0 6
            // at depth 2
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            if (dst & enPassentTarget)
            {
                addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
            }
#endif
            

            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

        while (myPawns)
        {
            uint64 pawn = getOne(myPawns);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) 
            {
                addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);

                if (dst) addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, bitScan(pawn));
            }

            // captures
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            dst = westCapture & enemyPieces;
            if (dst) addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);

            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = eastCapture & enemyPieces;
            if (dst) addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);

            // en-passent 
            // there can be only a single en-passent capture per pawn
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) 
            {
                // if the enPassent captured piece, the pawn and the king all lie in the same line, 
                // we need to check if the enpassent would move the king into check!!
                // really painful condition!!@!
                uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                     (pos->kings & myPieces);
                if (!causesCheck)
                {
                    addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
                }
            }
#endif

            myPawns ^= pawn;
        }

        // generate castling moves
        if (chance == WHITE)
        {
            if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F1G1 & allPieces) &&                          // squares between king and rook are empty
                !(F1G1 & threatened))                           // and not in threat from enemy pieces
            {
                // white king side castle
                addCastleMove(&nMoves, &newPositions, pos, BIT(E1), BIT(G1), BIT(H1), BIT(F1), chance);
            }
            if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B1D1 & allPieces) &&                          // squares between king and rook are empty
                !(C1D1 & threatened))                           // and not in threat from enemy pieces
            {
                // white queen side castle
                addCastleMove(&nMoves, &newPositions, pos, BIT(E1), BIT(C1), BIT(A1), BIT(D1), chance);
            }
        }
        else
        {
            if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F8G8 & allPieces) &&                          // squares between king and rook are empty
                !(F8G8 & threatened))                           // and not in threat from enemy pieces
            {
                // black king side castle
                addCastleMove(&nMoves, &newPositions, pos, BIT(E8), BIT(G8), BIT(H8), BIT(F8), chance);
            }
            if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B8D8 & allPieces) &&                          // squares between king and rook are empty
                !(C8D8 & threatened))                           // and not in threat from enemy pieces
            {
                // black queen side castle
                addCastleMove(&nMoves, &newPositions, pos, BIT(E8), BIT(C8), BIT(A8), BIT(D8), chance);
            }
        }
        
        // generate king moves
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(myKing);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while(kingMoves)
        {
            uint64 dst = getOne(kingMoves);
            addKingMove(&nMoves, &newPositions, pos, myKing, dst, chance);            
            kingMoves ^= dst;
        }

        // generate knight moves (only non-pinned knights can move)
        uint64 myKnights = (pos->knights & myPieces) & ~pinned;
        while (myKnights)
        {
            uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
            uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & ~myPieces;
#else
            uint64 knightMoves = knightAttacks(knight) & ~myPieces;
#endif
            while (knightMoves)
            {
                uint64 dst = getOne(knightMoves);
                addKnightMove(&nMoves, &newPositions, pos, knight, dst, chance);            
                knightMoves ^= dst;
            }
            myKnights ^= knight;
        }



        // generate bishop (and queen) moves
        uint64 myBishops = pos->bishopQueens & myPieces;

        // first deal with pinned bishops
        uint64 bishops = myBishops & pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            // TODO: bishopAttacks() function uses a kogge-stone sliding move generator. Switch to magics!
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
            bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);    // pined sliding pieces can move only along the line

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
                addSlidingMove(&nMoves, &newPositions, pos, bishop, dst, chance);            
                bishopMoves ^= dst;
            }
            bishops ^= bishop;
        }

        // remaining bishops/queens
        bishops = myBishops & ~pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
                addSlidingMove(&nMoves, &newPositions, pos, bishop, dst, chance);            
                bishopMoves ^= dst;
            }
            bishops ^= bishop;

        }


        // rook/queen moves
        uint64 myRooks = pos->rookQueens & myPieces;

        // first deal with pinned rooks
        uint64 rooks = myRooks & pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
            rookMoves &= sqsInLine(bitScan(rook), kingIndex);    // pined sliding pieces can move only along the line

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
                addSlidingMove(&nMoves, &newPositions, pos, rook, dst, chance);            
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }
        
        // remaining rooks/queens
        rooks = myRooks & ~pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
                addSlidingMove(&nMoves, &newPositions, pos, rook, dst, chance);            
                rookMoves ^= dst;
            }
            rooks ^= rook;

        }


        return nMoves;
    }




#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance>
#endif
    CUDA_CALLABLE_MEMBER __forceinline static uint32 generateMovesOutOfCheck (HexaBitBoardPosition *pos, CMove *genMoves,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened, 
                                           uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance
#endif
                                           )
    {
        uint32 nMoves = 0;
        uint64 king = pos->kings & myPieces;

        // figure out the no. of attackers 
        uint64 attackers = 0;

        // pawn attacks
        uint64 enemyPawns = allPawns & enemyPieces;
        attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) :
                                          (southEastOne(king) | southWestOne(king)) ) & enemyPawns;

        // knight attackers
        uint64 enemyKnights = pos->knights & enemyPieces;
        attackers |= knightAttacks(king) & enemyKnights;

        // bishop attackers
        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;

        // rook attackers
        uint64 enemyRooks = pos->rookQueens & enemyPieces;
        attackers |= rookAttacks(king, ~allPieces) & enemyRooks;


        // A. Try king moves to get the king out of check
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(king);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while(kingMoves)
        {
            uint64 dst = getOne(kingMoves);

            // TODO: set capture flag correctly
            addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), 0);
            kingMoves ^= dst;
        }


        // B. try moves to kill/block attacking pieces
        if (isSingular(attackers))
        {
            // Find the safe squares - i.e, if a dst square of a move is any of the safe squares, 
            // it will take king out of check

            // for pawn and knight attack, the only option is to kill the attacking piece
            // for bishops rooks and queens, it's the line between the attacker and the king, including the attacker
            uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
            
            // pieces that are pinned don't have any hope of saving the king
            // TODO: Think more about it
            myPieces &= ~pinned;

            // 1. pawn moves
            uint64 myPawns = allPawns & myPieces;

            // checking rank for pawn double pushes
            uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

            uint64 enPassentTarget = 0;
            if (pos->enPassent)
            {
                if (chance == BLACK)
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
                }
                else
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
                }
            }

            // en-passent can only save the king if the piece captured is the attacker
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
            if (enPassentCapturedPiece != attackers)
                enPassentTarget = 0;

            while (myPawns)
            {
                uint64 pawn = getOne(myPawns);

                // pawn push
                uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
                if (dst) 
                {
                    if (dst & safeSquares)
                    {
                        addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, 0);
                    }
                    else
                    {
                        // double push (only possible if single push was possible and single push didn't save the king)
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);

                        if (dst) 
                        {
                            addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
                        }
                    }
                }

                // captures (only one of the two captures will save the king.. if at all it does)
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst) 
                {
                    addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);
                }

                // en-passent 
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst) 
                {
                    addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(dst), CM_FLAG_EP_CAPTURE);
                }

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (pos->knights & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
                uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & safeSquares;
#else
                uint64 knightMoves = knightAttacks(knight) & safeSquares;
#endif
                while (knightMoves)
                {
                    uint64 dst = getOne(knightMoves);
                    // TODO: set capture flag correctly
                    addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), 0);
                    knightMoves ^= dst;
                }
                myKnights ^= knight;
            }
            
            // 3. bishop moves
            uint64 bishops = pos->bishopQueens & myPieces;
            while (bishops)
            {
                uint64 bishop = getOne(bishops);
                uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;

                while (bishopMoves)
                {
                    uint64 dst = getOne(bishopMoves);
                    // TODO: set capture flag correctly
                    addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), 0);
                    bishopMoves ^= dst;
                }
                bishops ^= bishop;
            }

            // 4. rook moves
            uint64 rooks = pos->rookQueens & myPieces;
            while (rooks)
            {
                uint64 rook = getOne(rooks);
                uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;

                while (rookMoves)
                {
                    uint64 dst = getOne(rookMoves);
                    // TODO: set capture flag correctly
                    addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), 0);
                    rookMoves ^= dst;
                }
                rooks ^= rook;
            }

        }   // end of if single attacker
        else
        {
            // multiple threats => only king moves possible
        }

        return nMoves;
    }


    // generates moves for the given board position
    // returns the no of moves generated
    // genMoves contains the generated moves
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER static uint32 generateMoves (HexaBitBoardPosition *pos, CMove *genMoves)
#else
    CUDA_CALLABLE_MEMBER static uint32 generateMoves (HexaBitBoardPosition *pos, CMove *genMoves, uint8 chance)
#endif
    {
        uint32 nMoves = 0;

        uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables

        uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
        uint64 blackPieces  = allPieces & (~pos->whitePieces);
        
        uint64 myPieces     = (chance == WHITE) ? pos->whitePieces : blackPieces;
        uint64 enemyPieces  = (chance == WHITE) ? blackPieces      : pos->whitePieces;

        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        uint64 enemyRooks   = pos->rookQueens & enemyPieces;

        uint64 myKing     = pos->kings & myPieces;
        uint8  kingIndex  = bitScan(myKing);

        uint64 pinned     = findPinnedPieces(pos->kings & myPieces, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);

        uint64 threatened = findAttackedSquares(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, 
                                                pos->knights & enemyPieces, pos->kings & enemyPieces, 
                                                myKing, !chance);



        // king is in check: call special generate function to generate only the moves that take king out of check
        if (threatened & (pos->kings & myPieces))
        {
#if USE_TEMPLATE_CHANCE_OPT == 1
            return generateMovesOutOfCheck<chance>(pos, genMoves, allPawns, allPieces, myPieces, enemyPieces, 
                                                              pinned, threatened, kingIndex);
#else
            return generateMovesOutOfCheck (pos, genMoves, allPawns, allPieces, myPieces, enemyPieces, 
                                            pinned, threatened, kingIndex, chance);
#endif
        }

        uint64 myPawns = allPawns & myPieces;

        // 0. generate en-passent moves first
        uint64 enPassentTarget = 0;
        if (pos->enPassent)
        {
            if (chance == BLACK)
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            }
            else
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
            }
        }
#if EN_PASSENT_GENERATION_NEW_METHOD == 1
        if (enPassentTarget)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);

            uint64 epSources = ((chance == WHITE) ? southEastOne(enPassentTarget) | southWestOne(enPassentTarget) : 
                                                    northEastOne(enPassentTarget) | northWestOne(enPassentTarget)) & myPawns;

            while (epSources)
            {
                uint64 pawn = getOne(epSources);
                if (pawn & pinned)
                {
                    // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
                    uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                    
                    if (enPassentTarget & line)
                    {
                        addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
                    }
                }
                else 
                /*if (!(enPassentCapturedPiece & pinned))*/   
                // the captured pawn should not be pinned in diagonal direction but it can be in vertical dir.
                // the diagonal pinning can't happen for enpassent in real chess game, so anyways it's not vaild
                {
                    uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                    uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                         (pos->kings & myPieces);
                    if (!causesCheck)
                    {
                        addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
                    }
                }
                epSources ^= pawn;
            }
        }
#endif
        // 1. pawn moves

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = sqsInLine(pawnIndex, kingIndex);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
            if (dst) 
            {
                addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), 0);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) 
                {
                    addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                addCompactPawnMoves(&nMoves, &genMoves, pawnIndex, dst, CM_FLAG_CAPTURE);
            }

            // en-passent capture isn't possible by a pinned pawn
            // TODO: think more about it
            // it's actually possible, if the pawn moves in the 'direction' of the pin
            // check out the position: rnb1kb1r/ppqp1ppp/2p5/4P3/2B5/6K1/PPP1N1PP/RNBQ3R b kq - 0 6
            // at depth 2
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            if (dst & enPassentTarget)
            {
                addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), CM_FLAG_EP_CAPTURE);
            }
#endif

            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

        while (myPawns)
        {
            uint64 pawn = getOne(myPawns);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) 
            {
                addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, 0);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);

                if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_DOUBLE_PAWN_PUSH);
            }

            // captures
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            dst = westCapture & enemyPieces;
            if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);

            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = eastCapture & enemyPieces;
            if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);

            // en-passent 
            // there can be only a single en-passent capture per pawn
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) 
            {
                // if the enPassent captured piece, the pawn and the king all lie in the same line, 
                // we need to check if the enpassent would move the king into check!!
                // really painful condition!!@!
                uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                     (pos->kings & myPieces);
                if (!causesCheck)
                {
                    addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_EP_CAPTURE);
                }
            }
#endif

            myPawns ^= pawn;
        }

        // generate castling moves
        if (chance == WHITE)
        {
            if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F1G1 & allPieces) &&                          // squares between king and rook are empty
                !(F1G1 & threatened))                           // and not in threat from enemy pieces
            {
                // white king side castle
                addCompactMove(&nMoves, &genMoves, E1, G1, CM_FLAG_KING_CASTLE);
            }
            if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B1D1 & allPieces) &&                          // squares between king and rook are empty
                !(C1D1 & threatened))                           // and not in threat from enemy pieces
            {
                // white queen side castle
                addCompactMove(&nMoves, &genMoves, E1, C1, CM_FLAG_QUEEN_CASTLE);
            }
        }
        else
        {
            if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F8G8 & allPieces) &&                          // squares between king and rook are empty
                !(F8G8 & threatened))                           // and not in threat from enemy pieces
            {
                // black king side castle
                addCompactMove(&nMoves, &genMoves, E8, G8, CM_FLAG_KING_CASTLE);
            }
            if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B8D8 & allPieces) &&                          // squares between king and rook are empty
                !(C8D8 & threatened))                           // and not in threat from enemy pieces
            {
                // black queen side castle
                addCompactMove(&nMoves, &genMoves, E8, C8, CM_FLAG_QUEEN_CASTLE);
            }
        }
        
        // generate king moves
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(myKing);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while(kingMoves)
        {
            uint64 dst = getOne(kingMoves);
            addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), 0); // TODO: correctly update capture flag
            kingMoves ^= dst;
        }

        // generate knight moves (only non-pinned knights can move)
        uint64 myKnights = (pos->knights & myPieces) & ~pinned;
        while (myKnights)
        {
            uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
            uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & ~myPieces;
#else
            uint64 knightMoves = knightAttacks(knight) & ~myPieces;
#endif
            while (knightMoves)
            {
                uint64 dst = getOne(knightMoves);
                addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), 0); // TODO: correctly update capture flag
                knightMoves ^= dst;
            }
            myKnights ^= knight;
        }



        // generate bishop (and queen) moves
        uint64 myBishops = pos->bishopQueens & myPieces;

        // first deal with pinned bishops
        uint64 bishops = myBishops & pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            // TODO: bishopAttacks() function uses a kogge-stone sliding move generator. Switch to magics!
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
            bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);    // pined sliding pieces can move only along the line

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
                addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), 0); // TODO: correctly update capture flag
                bishopMoves ^= dst;
            }
            bishops ^= bishop;
        }

        // remaining bishops/queens
        bishops = myBishops & ~pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
                addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), 0); // TODO: correctly update capture flag
                bishopMoves ^= dst;
            }
            bishops ^= bishop;

        }


        // rook/queen moves
        uint64 myRooks = pos->rookQueens & myPieces;

        // first deal with pinned rooks
        uint64 rooks = myRooks & pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
            rookMoves &= sqsInLine(bitScan(rook), kingIndex);    // pined sliding pieces can move only along the line

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
                addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), 0); // TODO: correctly update capture flag
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }
        
        // remaining rooks/queens
        rooks = myRooks & ~pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
                addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), 0); // TODO: correctly update capture flag
                rookMoves ^= dst;
            }
            rooks ^= rook;

        }


        return nMoves;
    }

#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance>
    CUDA_CALLABLE_MEMBER __forceinline static void makeMove (HexaBitBoardPosition *pos, CMove move)
#else
    CUDA_CALLABLE_MEMBER __forceinline static void makeMove (HexaBitBoardPosition *pos, CMove move, uint8 chance)
#endif
    {
        uint64 src = BIT(move.getFrom());
        uint64 dst = BIT(move.getTo());

        // figure out the source piece
        uint64 queens = pos->bishopQueens & pos->rookQueens;
        uint8 piece = 0;
        if (pos->kings & src)
            piece = KING;
        else if (pos->knights & src)
            piece = KNIGHT;
        else if ((pos->pawns & RANKS2TO7) & src)
            piece = PAWN;
        else if (queens & src)
            piece = QUEEN;
        else if (pos->bishopQueens & src)
            piece = BISHOP;
        else
            piece = ROOK;


        // promote the pawn (if this was promotion move)
        if (move.getFlags() == CM_FLAG_KNIGHT_PROMOTION || move.getFlags() == CM_FLAG_KNIGHT_PROMO_CAP)
            piece = KNIGHT;
        else if (move.getFlags() == CM_FLAG_BISHOP_PROMOTION || move.getFlags() == CM_FLAG_BISHOP_PROMO_CAP)
            piece = BISHOP;
        else if (move.getFlags() == CM_FLAG_ROOK_PROMOTION || move.getFlags() == CM_FLAG_ROOK_PROMO_CAP)
            piece = ROOK;
        else if (move.getFlags() == CM_FLAG_QUEEN_PROMOTION || move.getFlags() == CM_FLAG_QUEEN_PROMO_CAP)
            piece = QUEEN;

        // remove source from all bitboards
        pos->bishopQueens &= ~src;
        pos->rookQueens   &= ~src;
        pos->kings        &= ~src;
        pos->knights      &= ~src;
        pos->pawns        &= ~(src & RANKS2TO7);

        // remove the dst from all bitboards
        pos->bishopQueens &= ~dst;
        pos->rookQueens   &= ~dst;
        pos->kings        &= ~dst;
        pos->knights      &= ~dst;
        pos->pawns        &= ~(dst & RANKS2TO7);

        // put the piece that moved in the required bitboards
        if (piece == KING)
        {
            pos->kings          |= dst;

            if (chance == WHITE)
                pos->whiteCastle = 0;
            else
                pos->blackCastle = 0;
        }

        if (piece == KNIGHT)
            pos->knights        |= dst;

        if (piece == PAWN)
            pos->pawns          |= dst;

        if (piece == BISHOP || piece == QUEEN)
            pos->bishopQueens   |= dst;

        if (piece == ROOK || piece == QUEEN)
            pos->rookQueens     |= dst;


        if (chance == WHITE)
        {
            pos->whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            pos->whitePieces  = pos->whitePieces  & ~dst;
        }

        // if it's an en-passet move, remove the captured pawn also
        if (move.getFlags() == CM_FLAG_EP_CAPTURE)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(dst) : northOne(dst);

            pos->pawns              &= ~(enPassentCapturedPiece & RANKS2TO7);

            if (chance == BLACK)
                pos->whitePieces    &= ~enPassentCapturedPiece;
        }

        // if it's a castling, move the rook also
        if (chance == WHITE)
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // white castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H1)) | BIT(F1);
                pos->whitePieces = (pos->whitePieces ^ BIT(H1)) | BIT(F1);
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // white castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A1)) | BIT(D1);
                pos->whitePieces = (pos->whitePieces ^ BIT(A1)) | BIT(D1);
            }
        }
        else
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // black castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H8)) | BIT(F8);
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // black castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A8)) | BIT(D8);
            }
        }


        // update the game state
        pos->chance = !chance;
        pos->enPassent = 0;
        //pos->halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture
        updateCastleFlag(pos, dst,  chance);

        if (piece == ROOK)
        {
            updateCastleFlag(pos, src, !chance);
        }

        if (move.getFlags() == CM_FLAG_DOUBLE_PAWN_PUSH)
        {
            pos->enPassent = (move.getFrom() & 7) + 1;      // store file + 1
        }
    }

#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance>
#endif
    CUDA_CALLABLE_MEMBER __forceinline static uint32 countMovesOutOfCheck (HexaBitBoardPosition *pos,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened, 
                                           uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance
#endif
                                           )
    {
        uint32 nMoves = 0;
        uint64 king = pos->kings & myPieces;

        // figure out the no. of attackers 
        uint64 attackers = 0;

        // pawn attacks
        uint64 enemyPawns = allPawns & enemyPieces;
        attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) :
                                          (southEastOne(king) | southWestOne(king)) ) & enemyPawns;

        // knight attackers
        uint64 enemyKnights = pos->knights & enemyPieces;
        attackers |= knightAttacks(king) & enemyKnights;

        // bishop attackers
        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;

        // rook attackers
        uint64 enemyRooks = pos->rookQueens & enemyPieces;
        attackers |= rookAttacks(king, ~allPieces) & enemyRooks;


        // A. Try king moves to get the king out of check
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(king);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        nMoves += popCount(kingMoves);

        // B. try moves to kill/block attacking pieces
        if (isSingular(attackers))
        {
            // Find the safe squares - i.e, if a dst square of a move is any of the safe squares, 
            // it will take king out of check

            // for pawn and knight attack, the only option is to kill the attacking piece
            // for bishops rooks and queens, it's the line between the attacker and the king, including the attacker
            uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
            
            // pieces that are pinned don't have any hope of saving the king
            // TODO: Think more about it
            myPieces &= ~pinned;

            // 1. pawn moves
            uint64 myPawns = allPawns & myPieces;

            // checking rank for pawn double pushes
            uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

            uint64 enPassentTarget = 0;
            if (pos->enPassent)
            {
                if (chance == BLACK)
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
                }
                else
                {
                    enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
                }
            }

            // en-passent can only save the king if the piece captured is the attacker
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
            if (enPassentCapturedPiece != attackers)
                enPassentTarget = 0;

            while (myPawns)
            {
                uint64 pawn = getOne(myPawns);

                // pawn push
                uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
                if (dst) 
                {
                    if (dst & safeSquares)
                    {
                        if (dst & (RANK1 | RANK8))
                            nMoves += 4;    // promotion
                        else
                            nMoves++;
                    }
                    else
                    {
                        // double push (only possible if single push was possible and single push didn't save the king)
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);

                        if (dst) 
                        {
                            nMoves++;
                        }
                    }
                }

                // captures (only one of the two captures will save the king.. if at all it does)
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst) 
                {
                    if (dst & (RANK1 | RANK8))
                        nMoves += 4;    // promotion
                    else
                        nMoves++;
                }

                // en-passent 
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst) 
                {
                    nMoves++;
                }

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (pos->knights & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
                uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & safeSquares;
#else
                uint64 knightMoves = knightAttacks(knight) & safeSquares;
#endif
                nMoves += popCount(knightMoves);
                myKnights ^= knight;
            }
            
            // 3. bishop moves
            uint64 bishops = pos->bishopQueens & myPieces;
            while (bishops)
            {
                uint64 bishop = getOne(bishops);
                uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;

                nMoves += popCount(bishopMoves);
                bishops ^= bishop;
            }

            // 4. rook moves
            uint64 rooks = pos->rookQueens & myPieces;
            while (rooks)
            {
                uint64 rook = getOne(rooks);
                uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;

                nMoves += popCount(rookMoves);
                rooks ^= rook;
            }

        }   // end of if single attacker
        else
        {
            // multiple threats => only king moves possible
        }

        return nMoves;
    }



    // count moves for the given board position
    // returns the no of moves generated
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER static uint32 countMoves (HexaBitBoardPosition *pos)
#else
    CUDA_CALLABLE_MEMBER static uint32 countMoves (HexaBitBoardPosition *pos, uint8 chance)
#endif
    {
        uint32 nMoves = 0;

        uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables

        uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
        uint64 blackPieces  = allPieces & (~pos->whitePieces);
        
        uint64 myPieces     = (chance == WHITE) ? pos->whitePieces : blackPieces;
        uint64 enemyPieces  = (chance == WHITE) ? blackPieces      : pos->whitePieces;

        uint64 enemyBishops = pos->bishopQueens & enemyPieces;
        uint64 enemyRooks   = pos->rookQueens & enemyPieces;

        uint64 myKing     = pos->kings & myPieces;
        uint8  kingIndex  = bitScan(myKing);

        uint64 pinned     = findPinnedPieces(pos->kings & myPieces, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);

        uint64 threatened = findAttackedSquares(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, 
                                                pos->knights & enemyPieces, pos->kings & enemyPieces, 
                                                myKing, !chance);


        // king is in check: call special generate function to generate only the moves that take king out of check
        if (threatened & (pos->kings & myPieces))
        {
#if USE_TEMPLATE_CHANCE_OPT == 1
            return countMovesOutOfCheck<chance>(pos, allPawns, allPieces, myPieces, enemyPieces, 
                                                              pinned, threatened, kingIndex);
#else
            return countMovesOutOfCheck (pos, allPawns, allPieces, myPieces, enemyPieces, 
                                         pinned, threatened, kingIndex, chance);
#endif
        }

        uint64 myPawns = allPawns & myPieces;

        // 0. generate en-passent moves first
        uint64 enPassentTarget = 0;
        if (pos->enPassent)
        {
            if (chance == BLACK)
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            }
            else
            {
                enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
            }
        }
#if EN_PASSENT_GENERATION_NEW_METHOD == 1
        if (enPassentTarget)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);

            uint64 epSources = ((chance == WHITE) ? southEastOne(enPassentTarget) | southWestOne(enPassentTarget) : 
                                                    northEastOne(enPassentTarget) | northWestOne(enPassentTarget)) & myPawns;

            while (epSources)
            {
                uint64 pawn = getOne(epSources);
                if (pawn & pinned)
                {
                    // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
                    uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                    
                    if (enPassentTarget & line)
                    {
                        nMoves++;
                    }
                }
                else 
                /*if (!(enPassentCapturedPiece & pinned))*/   
                // the captured pawn should not be pinned in diagonal direction but it can be in vertical dir.
                // the diagonal pinning can't happen for enpassent in real chess game, so anyways it's not vaild
                {
                    uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                    uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                         (pos->kings & myPieces);
                    if (!causesCheck)
                    {
                        nMoves++;
                    }
                }
                epSources ^= pawn;
            }
        }
#endif
        // 1. pawn moves

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = sqsInLine(pawnIndex, kingIndex);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
            if (dst) 
            {
                nMoves++;

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) 
                {
                    nMoves++;
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                if (dst & (RANK1 | RANK8))
                    nMoves += 4;    // promotion
                else
                    nMoves++;
            }

            // en-passent capture isn't possible by a pinned pawn
            // TODO: think more about it
            // it's actually possible, if the pawn moves in the 'direction' of the pin
            // check out the position: rnb1kb1r/ppqp1ppp/2p5/4P3/2B5/6K1/PPP1N1PP/RNBQ3R b kq - 0 6
            // at depth 2
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            if (dst & enPassentTarget)
            {
                nMoves++;
            }
#endif
            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

        // pawn push
        uint64 dsts = ((chance == WHITE) ? northOne(myPawns) : southOne(myPawns)) & (~allPieces);
        nMoves += popCount(dsts);
        uint64 promotions = dsts & (RANK1 | RANK8);
        nMoves += 3 * popCount(promotions);

        // double push
        dsts = ((chance == WHITE) ? northOne(dsts & checkingRankDoublePush): 
                                    southOne(dsts & checkingRankDoublePush) ) & (~allPieces);
        nMoves += popCount(dsts);

        // captures
        dsts = ((chance == WHITE) ? northWestOne(myPawns) : southWestOne(myPawns)) & enemyPieces;
        nMoves += popCount(dsts);
        promotions = dsts & (RANK1 | RANK8);
        nMoves += 3 * popCount(promotions);


        dsts = ((chance == WHITE) ? northEastOne(myPawns) : southEastOne(myPawns)) & enemyPieces;
        nMoves += popCount(dsts);
        promotions = dsts & (RANK1 | RANK8);
        nMoves += 3 * popCount(promotions);

        // generate castling moves
        if (chance == WHITE)
        {
            if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F1G1 & allPieces) &&                          // squares between king and rook are empty
                !(F1G1 & threatened))                           // and not in threat from enemy pieces
            {
                // white king side castle
                nMoves++;
            }
            if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B1D1 & allPieces) &&                          // squares between king and rook are empty
                !(C1D1 & threatened))                           // and not in threat from enemy pieces
            {
                // white queen side castle
                nMoves++;
            }
        }
        else
        {
            if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F8G8 & allPieces) &&                          // squares between king and rook are empty
                !(F8G8 & threatened))                           // and not in threat from enemy pieces
            {
                // black king side castle
                nMoves++;
            }
            if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B8D8 & allPieces) &&                          // squares between king and rook are empty
                !(C8D8 & threatened))                           // and not in threat from enemy pieces
            {
                // black queen side castle
                nMoves++;
            }
        }
        
        // generate king moves
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(myKing);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        nMoves += popCount(kingMoves);

        // generate knight moves (only non-pinned knights can move)
        uint64 myKnights = (pos->knights & myPieces) & ~pinned;
        while (myKnights)
        {
            uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
            uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & ~myPieces;
#else
            uint64 knightMoves = knightAttacks(knight) & ~myPieces;
#endif
            nMoves += popCount(knightMoves);
            myKnights ^= knight;
        }


        // generate bishop (and queen) moves
        uint64 myBishops = pos->bishopQueens & myPieces;

        // first deal with pinned bishops
        uint64 bishops = myBishops & pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            // TODO: bishopAttacks() function uses a kogge-stone sliding move generator. Switch to magics!
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
            bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);    // pined sliding pieces can move only along the line

            nMoves += popCount(bishopMoves);
            bishops ^= bishop;
        }

        // remaining bishops/queens
        bishops = myBishops & ~pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;

            nMoves += popCount(bishopMoves);
            bishops ^= bishop;

        }

        // rook/queen moves
        uint64 myRooks = pos->rookQueens & myPieces;

        // first deal with pinned rooks
        uint64 rooks = myRooks & pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
            rookMoves &= sqsInLine(bitScan(rook), kingIndex);    // pined sliding pieces can move only along the line

            nMoves += popCount(rookMoves);
            rooks ^= rook;
        }
        
        // remaining rooks/queens
        rooks = myRooks & ~pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;

            nMoves += popCount(rookMoves);
            rooks ^= rook;
        }

        return nMoves;
    }
};


// random generators and basic idea of finding magics taken from:
// http://chessprogramming.wikispaces.com/Looking+for+Magics 

uint64 random_uint64() 
{
      uint64 u1, u2, u3, u4;
      u1 = (uint64)(rand()) & 0xFFFF; u2 = (uint64)(rand()) & 0xFFFF;
      u3 = (uint64)(rand()) & 0xFFFF; u4 = (uint64)(rand()) & 0xFFFF;
      return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
}

uint64 random_uint64_sparse() 
{
    return random_uint64() & random_uint64() & random_uint64();
} 

// get i'th combo mask 
uint64 getOccCombo(uint64 mask, uint64 i)
{
    uint64 op = 0;
    while(i)
    {
        int bit = i % 2;
        uint64 opBit = MoveGeneratorBitboard::getOne(mask);
        mask &= ~opBit;
        op |= opBit * bit;
        i = i >> 1;
    }

    return op;
}

uint64 findMagicCommon(uint64 occCombos[], uint64 attacks[], uint64 attackTable[], int numCombos, int bits, uint64 preCalculatedMagic = 0)
{
    uint64 magic = 0;
    while(1)
    {
        if (preCalculatedMagic)
        {
            magic = preCalculatedMagic;
        }
        else
        {
            for (int i=0; i < (1 << bits); i++)
            {
                attackTable[i] = 0; // unused entry
            }

            magic = random_uint64_sparse();
            //magic = random_uint64();
        }

        // try all possible occupancy combos and check for collisions
        int i = 0;
        for (i = 0; i < numCombos; i++)
        {
            uint64 index = (magic * occCombos[i]) >> (64 - bits);
            if (preCalculatedMagic || attackTable[index] == 0)
            {
                attackTable[index] = attacks[i];
            }
            else
            {
                // mismatching entry already found
                if (attackTable[index] != attacks[i])
                    break;
            }
        }

        if (i == numCombos)
            break;
        else
            assert(preCalculatedMagic == 0);
    }
    return magic;
}

uint64 findRookMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic)
{
    uint64 mask = RookAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];     // the occupancy bits for each combination (actually permutation)
    uint64 attacks[4096];       // attacks for every combo (calculated using kogge stone)

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = MoveGeneratorBitboard::rookAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, ROOK_MAGIC_BITS, preCalculatedMagic);

}

uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic)
{
    uint64 mask = BishopAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];     // the occupancy bits for each combination (actually permutation)
    uint64 attacks[4096];       // attacks for every combo (calculated using kogge stone)

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = MoveGeneratorBitboard::bishopAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, BISHOP_MAGIC_BITS, preCalculatedMagic);
}

// only for testing
#if 0
uint64 rookMagicAttackTables[64][1 << ROOK_MAGIC_BITS];
uint64 bishopMagicAttackTables[64][1 << BISHOP_MAGIC_BITS];

void findBishopMagics()
{
    printf("\n\nBishop Magics: ...");
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findBishopMagicForSquare(square, bishopMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }

}

void findRookMagics()
{

    printf("\n\nRook Magics: ...");
    //int square = A8;
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findRookMagicForSquare(square, rookMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }

}


void findMagics()
{
    srand (time(NULL));

    findBishopMagics();
    findRookMagics();
}
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
        HexaBitBoardPosition    *allChildBoards;
        CMove                   *allSecondLevelChildMoves;
        HexaBitBoardPosition   **boardPointers;
        uint32                  *allChildCounters;
    };
};

__launch_bounds__( BLOCK_SIZE, 4 )
__global__ void perft_bb_gpu_single_level(HexaBitBoardPosition *position, uint64 *globalPerftCounter, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos = position[index];

    uint8 color = pos.chance;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

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
__launch_bounds__( BLOCK_SIZE, 4 )
__global__ void makeMove_and_perft_single_level(HexaBitBoardPosition **positions, CMove *generatedMoves, uint64 *globalPerftCounter, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    HexaBitBoardPosition *posPointer = positions[index];
    HexaBitBoardPosition pos = *posPointer;
    uint8 color = pos.chance;

    CMove move = generatedMoves[index];

    // 1. make the move
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(&pos, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(&pos, move);
    }
#else
        MoveGeneratorBitboard::makeMove(&pos, move, color);
#endif


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
__launch_bounds__( BLOCK_SIZE, 4 )
__global__ void count_moves_single_level(HexaBitBoardPosition *position, uint32 *moveCounts, int nThreads)
{

    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos = position[index];

    uint8 color = pos.chance;

    // just count the no. of moves for each board and save it in moveCounts array
    int nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

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

// generatedMoves is array of pointers
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

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(&pos, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(&pos, move);
    }
#else
        MoveGeneratorBitboard::makeMove(&pos, move, chance);
#endif

    positions[index] = pos;
}
#endif

// this version launches two levels as a single gird
// to be called only at depth == 3
// ~20 Billion moves per second in best case!
__global__ void perft_bb_gpu_depth3(HexaBitBoardPosition *position, uint64 *globalPerftCounter, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos = position[index];

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    uint8 color = pos.chance;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 allMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (allMoves == 0)
        return;

    // first thread of block allocates memory to store all moves generated by the thread block
    if (threadIdx.x == 0)
    {
        //printf("\nFirst level moves: %d\n", allMoves);
        int hr;
        hr = deviceMalloc(&shMem.allChildBoards, sizeof(HexaBitBoardPosition) * allMoves);
        /*
        if (hr != 0)
            printf("error in malloc for childBoards at depth %d, for %d moves\n", 3, allMoves);
        else
            printf("\nAllocated allChildBoards of %d bytes, address: %X\n", sizeof(HexaBitBoardPosition) * allMoves, shMem.allChildBoards);
        */
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    HexaBitBoardPosition *childBoards = shMem.allChildBoards + moveListOffset;

    // 3. generate the moves now
    if (nMoves)
    {
        generateBoards(&pos, color, childBoards);
    }

    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    cudaStream_t childStream;
    uint32 *childMoveCounts;

    if (threadIdx.x == 0)
    {
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        uint32 nBlocks = (allMoves - 1) / BLOCK_SIZE + 1;
        {
            int hr = deviceMalloc(&childMoveCounts, sizeof(uint32) * allMoves);
            /*
            if (hr != 0)
                printf("error in malloc for childMoveCounts at depth %d, for %d moves\n", 3, allMoves);
            else
                printf("\nAllocated childMoveCounts of %d bytes, address: %X\n", sizeof(uint32) * allMoves, childMoveCounts);
            */

            shMem.allChildCounters = childMoveCounts;

            // first count the moves that would be generated by childs
            count_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(childBoards, childMoveCounts, allMoves);
            cudaDeviceSynchronize();
        }
    }

    __syncthreads();

    childMoveCounts = shMem.allChildCounters;

    // movelistOffset contains starting location of childs for each thread
    uint32 localMoveCounter = 0;    // no. of child moves generated by child boards of this thread
    for (int i = 0; i < nMoves; i++)
    {
        localMoveCounter += childMoveCounts[moveListOffset + i];
    }

    // put localMoveCounter in shared memory and perform a scan to get first level scan
    shMem.movesForThread[threadIdx.x] = localMoveCounter;

    __syncthreads();
    scan(shMem.movesForThread);

    uint32 allSecondLevelMoves = shMem.movesForThread[blockDim.x - 1];
    if (allSecondLevelMoves == 0)
    {
        if (threadIdx.x == 0)
        {
            deviceFree(childBoards);
            deviceFree(childMoveCounts);
            cudaStreamDestroy(childStream);
        }
        return;
    }

    // first thread of the block allocates memory for all second level childs
    if (threadIdx.x == 0)
    {
        /*
        printf("\nSecond level moves: %d, per thread breakup:\n", allSecondLevelMoves);
        for (int i=0;i < blockDim.x; i++)
            printf("% d", shMem.movesForThread[i]);
        */

        int hr;
        hr = deviceMalloc(&shMem.allSecondLevelChildMoves, sizeof(CMove) * allSecondLevelMoves);
        /*
        if (hr != 0)
            printf("error in malloc for allSecondLevelChildMoves at depth %d, for %d moves\n", 3, allSecondLevelMoves);
        else
            printf("\nAllocated allSecondLevelChildMoves of %d bytes, address: %X\n", sizeof(CMove) * allSecondLevelMoves, shMem.allSecondLevelChildMoves);
        */

        hr = deviceMalloc(&shMem.boardPointers, sizeof(void *) * allSecondLevelMoves);
        /*
        if (hr != 0)
            printf("error in malloc for boardPointers at depth %d, for %d moves\n", 3, allSecondLevelMoves);
        else
            printf("\nAllocated boardPointers of %d bytes, address: %X\n", sizeof(void *) * allSecondLevelMoves, shMem.boardPointers);
        */
    }

    __syncthreads();
    CMove *secondLevelChildMoves = shMem.allSecondLevelChildMoves;

    // do full scan of childMoveCounts global memory array to get starting offsets of all second level child moves
    // all threads do this in a co-operative way
    uint32 baseOffset = shMem.movesForThread[threadIdx.x] - localMoveCounter;
    HexaBitBoardPosition **boardPointers = shMem.boardPointers;

#if USE_COLAESED_WRITES_FOR_MOVELIST_SCAN == 1
    // this is only a very little bit faster than (much) simpler approach below
    // maybe try interval-expand algorithm (http://nvlabs.github.io/moderngpu/intervalmove.html)

    uint32 laneIdx = threadIdx.x & 0x1F;
    for (int curLane = 0; curLane < WARP_SIZE; curLane++)
    {
        int curNMoves         =           __shfl((int) nMoves, curLane);
        int curMoveListOffset =           __shfl((int) moveListOffset, curLane);
        int curBaseOffset     =           __shfl((int) baseOffset, curLane);
        HexaBitBoardPosition *curChildBoards = (HexaBitBoardPosition *) __shfl((int) childBoards, curLane);
        for (int i = 0; i < curNMoves; i++)
        {
            int nChildMoves;
            if (laneIdx == 0)
            {
                nChildMoves         = childMoveCounts[curMoveListOffset + i];
                childMoveCounts[curMoveListOffset + i] = (uint32) (secondLevelChildMoves + curBaseOffset);
            }
            nChildMoves = __shfl(nChildMoves, 0);

            // uint32 nChildMoves = childMoveCounts[curMoveListOffset + i];
            HexaBitBoardPosition *currentBoardPointer = &curChildBoards[i];

            uint32 curOffset = 0;
            while (curOffset < nChildMoves)
            {
                if (curOffset + laneIdx < nChildMoves)
                    boardPointers[curBaseOffset + curOffset + laneIdx] = currentBoardPointer;

                curOffset += WARP_SIZE;
            }

            curBaseOffset += nChildMoves;
        }
    }

#else
    // TODO: this operation is expensive
    // fix this by colaesing memory reads/writes

    for (int i = 0; i < nMoves; i++)
    {
        uint32 nChildMoves = childMoveCounts[moveListOffset + i];
        HexaBitBoardPosition *currentBoardPointer = &childBoards[i];

        for (int j=0; j < nChildMoves; j++)
        {
            // this is about 2000 writes for each thread!
            boardPointers[baseOffset + j] = currentBoardPointer;
        }

        childMoveCounts[moveListOffset + i] = (uint32) (secondLevelChildMoves + baseOffset);
        baseOffset += nChildMoves;
    }
#endif

    __syncthreads();
    // childMoveCounts now have the exclusive scan - containing the addresses to put moves on

    if (threadIdx.x == 0)
    {
        // first thread of the block now launches kernel that generates second level moves

        uint32 nBlocks = (allMoves - 1) / BLOCK_SIZE + 1;
        generate_moves_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>>(childBoards, (CMove **) childMoveCounts, allMoves);

        // now we have all second level generated moves in secondLevelChildBoards .. launch a kernel at depth - 2 to make the moves and count leaves
        nBlocks = (allSecondLevelMoves - 1) / BLOCK_SIZE + 1;

        makeMove_and_perft_single_level<<<nBlocks, BLOCK_SIZE, 0, childStream>>> (boardPointers, secondLevelChildMoves, globalPerftCounter, allSecondLevelMoves);

        // when preallocated memory is used, we don't need to free memory
        // which also means that there is no need to wait for child kernel to finish
#if USE_PREALLOCATED_MEMORY != 1
        cudaDeviceSynchronize();

        //printf("\nFreeing childBoards: %X\n", childBoards);
        deviceFree(childBoards);
        //printf("\nFreeing childMoveCounts: %X\n", childMoveCounts);
        deviceFree(childMoveCounts);
        //printf("\nFreeing secondLevelChildMoves: %X\n", secondLevelChildMoves);
        deviceFree(secondLevelChildMoves);
        //printf("\nFreeing boardPointers: %X\n", boardPointers);
        deviceFree(boardPointers);
#endif
        cudaStreamDestroy(childStream);
    }
}


// this version processes one level a time until it reaches depth 4 - where perft_bb_gpu_depth3 is called
// DON'T CALL this with DEPTH = 1 (call perft_bb_gpu_single_level instead)
__global__ void perft_bb_gpu_safe(HexaBitBoardPosition *position, uint64 *globalPerftCounter, int depth, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    HexaBitBoardPosition pos = position[index];

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    uint8 color = pos.chance;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 allMoves = shMem.movesForThread[blockDim.x - 1];

    // nothing more to do!
    if (allMoves == 0)
        return;

    if (threadIdx.x == 0)
    {
        int hr;
        hr = deviceMalloc(&shMem.allChildBoards, sizeof(HexaBitBoardPosition) * allMoves);
        /*
        if (hr != 0)
            printf("error in malloc for childBoards at depth %d, for %d moves\n", depth, allMoves);
        */
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    HexaBitBoardPosition *childBoards = shMem.allChildBoards + moveListOffset;

    // 3. generate the moves now
    if (nMoves)
    {
        generateBoards(&pos, color, childBoards);
    }

    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    if (threadIdx.x == 0)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        uint32 nBlocks = (allMoves - 1) / BLOCK_SIZE + 1;

        if (depth == 2)
            perft_bb_gpu_single_level<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (childBoards, globalPerftCounter, allMoves);
        else if (depth == 4)
            perft_bb_gpu_depth3<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (childBoards, globalPerftCounter, allMoves);
        else
            perft_bb_gpu_safe<<<nBlocks, BLOCK_SIZE, sizeof(sharedMemAllocs), childStream>>> (childBoards, globalPerftCounter, depth-1, allMoves);

        // when preallocated memory is used, we don't need to free memory
        // which also means that there is no need to wait for child kernel to finish
#if USE_PREALLOCATED_MEMORY != 1
        cudaDeviceSynchronize();
        deviceFree(childBoards);
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
        perft_bb_gpu_safe<<<1, BLOCK_SIZE, sizeof(sharedMemAllocs), 0>>> (pos, globalPerftCounter, depth, 1);
        cudaDeviceSynchronize();

        // 'free' up the memory used by the launch
        preAllocatedMemoryUsed = 0;
    }
    else
    {
        // recurse serially till we reach a depth where we can launch parallel work
        nMoves = generateBoards(pos, color, boardStack);
        for (uint32 i=0; i < nMoves; i++)
        {
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
