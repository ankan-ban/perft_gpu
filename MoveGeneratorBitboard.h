#include "chess.h"
#include <intrin.h>

#define DEBUG_PRINT_MOVES 0
#if DEBUG_PRINT_MOVES == 1
    #define DEBUG_PRINT_DEPTH 6
    bool printMoves = false;
#endif

// use trove library coalesced_ptr (that makes use of SHFL instruction) to optimize AOS reads
#define USE_TROVE_AOS_OPT 0
#if USE_TROVE_AOS_OPT == 1
#include <trove/ptr.h>
#endif

// only count moves at leaves (instead of generating/making them)
#define USE_COUNT_ONLY_OPT true

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
#define USE_KING_LUT 0

// use lookup table for knight moves
#define USE_KNIGHT_LUT 0

// use lookup table (magics) for sliding moves (TODO)
#define USE_SLIDING_LUT 0

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

#define DIAGONAL_A1H8  C64(0x8040201008040201)
#define DIAGONAL_A8H1  C64(0x0102040810204080)

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

#if TEST_GPU_PERFT == 1
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
#endif

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


class MoveGeneratorBitboard
{
private:

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

    CUDA_CALLABLE_MEMBER __forceinline static uint64 bishopAttacks(uint64 bishops, uint64 pro)
    {
        return northEastAttacks(bishops, pro) |
               northWestAttacks(bishops, pro) |
               southEastAttacks(bishops, pro) |
               southWestAttacks(bishops, pro) ;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 rookAttacks(uint64 rooks, uint64 pro)
    {
        return northAttacks(rooks, pro) |
               southAttacks(rooks, pro) |
               eastAttacks (rooks, pro) |
               westAttacks (rooks, pro) ;
    }

    CUDA_CALLABLE_MEMBER __forceinline static uint64 queenAttacks(uint64 queens, uint64 pro)
    {
        return rookAttacks  (queens, pro) |
               bishopAttacks(queens, pro) ;
    }

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


public:

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
        line  =      (   (file  & 0xff) - 1) & a2a7; // a2a7 if same file
        line += 2 * ((   (rank  & 0xff) - 1) >> 58); // b1g1 if same rank
        line += (((rank - file) & 0xff) - 1) & b2g7; // b2g7 if same diagonal
        line += (((rank + file) & 0xff) - 1) & h1b7; // h1b7 if same antidiag
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
        return squaresInBetween(min(sq1, sq2), max(sq1, sq2));
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
#if TEST_GPU_PERFT == 1
        // copy all the lookup tables from CPU's memory to GPU memory
        cudaError_t err = cudaMemcpyToSymbol(gBetween, Between, sizeof(Between));
        printf("For copying between table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

        err = cudaMemcpyToSymbol(gLine, Line, sizeof(Line));
        printf("For copying line table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gRookAttacks, RookAttacks, sizeof(RookAttacks));
        printf("For copying RookAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gBishopAttacks, BishopAttacks, sizeof(BishopAttacks));
        printf("For copying BishopAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  
        
        err = cudaMemcpyToSymbol(gQueenAttacks, QueenAttacks, sizeof(QueenAttacks));
        printf("For copying QueenAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gKnightAttacks, KnightAttacks, sizeof(KnightAttacks));
        printf("For copying KnightAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  

        err = cudaMemcpyToSymbol(gKingAttacks, KingAttacks, sizeof(KingAttacks));
        printf("For copying KingAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));  
#endif

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
        attacked |= bishopAttacks(enemyBishops, emptySquares | myKing); // squares behind king are also under threat (in the sense that king can't go there)

        // 4. rook attacks
        attacked |= rookAttacks(enemyRooks, emptySquares | myKing); // squares behind king are also under threat

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

#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance, bool countOnly>
#endif
    CUDA_CALLABLE_MEMBER __forceinline static uint32 generateMovesOutOfCheck (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened, 
                                           uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance, bool countOnly
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
        if (countOnly)
        {
            nMoves += popCount(kingMoves);
        } 
        else
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
                        if (countOnly) 
                        {
                            if (dst & (RANK1 | RANK8))
                                nMoves += 4;    // promotion
                            else
                                nMoves++;
                        }
                        else addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
                    }
                    else
                    {
                        // double push (only possible if single push was possible and single push didn't save the king)
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);

                        if (dst) 
                        {
                            if (countOnly) nMoves++;
                            else addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, bitScan(pawn));
                        }
                    }
                }

                // captures (only one of the two captures will save the king.. if at all it does)
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst) 
                {
                    if (countOnly) 
                    {
                        if (dst & (RANK1 | RANK8))
                            nMoves += 4;    // promotion
                        else
                            nMoves++;
                    }
                    else addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
                }

                // en-passent 
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst) 
                {
                    if (countOnly) nMoves++;
                    else addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
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
                if (countOnly) 
                {
                    nMoves += popCount(knightMoves);
                }
                else
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

                if (countOnly) 
                {
                    nMoves += popCount(bishopMoves);
                }
                else
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

                if (countOnly) 
                {
                    nMoves += popCount(rookMoves);
                }
                else
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
    template <uint8 chance, bool countOnly>
    CUDA_CALLABLE_MEMBER static uint32 generateMoves (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions)
#else
    CUDA_CALLABLE_MEMBER static uint32 generateMoves (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions, uint8 chance, bool countOnly)
#endif
    {
        uint32 nMoves = 0;

        //uint8 chance = pos->chance;

        // TODO: implement fast path for count only
        // might be better to do it either in a seperate function.. or make this function templated on countOnly
        // bool countOnly = (newPositions == NULL);

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
            return generateMovesOutOfCheck<chance, countOnly>(pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, 
                                                              pinned, threatened, kingIndex);
#else
            return generateMovesOutOfCheck (pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, 
                                            pinned, threatened, kingIndex, chance, countOnly);
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
                        if (countOnly) nMoves++;
                        else addEnPassentMove(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
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
                        if (countOnly) nMoves++;
                        else addEnPassentMove(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
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
                if (countOnly) nMoves++;
                else addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, false, pawnIndex);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) 
                {
                    if (countOnly) nMoves++;
                    else addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, pawnIndex);
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                if (countOnly) {
                    if (dst & (RANK1 | RANK8))
                        nMoves += 4;    // promotion
                    else
                        nMoves++;
                }
                else 
                {
                    addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);
                }
            }

            // en-passent capture isn't possible by a pinned pawn
            // TODO: think more about it
            // it's actually possible, if the pawn moves in the 'direction' of the pin
            // check out the position: rnb1kb1r/ppqp1ppp/2p5/4P3/2B5/6K1/PPP1N1PP/RNBQ3R b kq - 0 6
            // at depth 2
#if EN_PASSENT_GENERATION_NEW_METHOD != 1
            if (dst & enPassentTarget)
            {
                if (countOnly) nMoves++;
                else addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
            }
#endif
            

            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;


        if (countOnly)
        {
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
        }
        else
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
                if (countOnly) nMoves++;
                else addCastleMove(&nMoves, &newPositions, pos, BIT(E1), BIT(G1), BIT(H1), BIT(F1), chance);
            }
            if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B1D1 & allPieces) &&                          // squares between king and rook are empty
                !(C1D1 & threatened))                           // and not in threat from enemy pieces
            {
                // white queen side castle
                if (countOnly) nMoves++;
                else addCastleMove(&nMoves, &newPositions, pos, BIT(E1), BIT(C1), BIT(A1), BIT(D1), chance);
            }
        }
        else
        {
            if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) &&   // castle flag is set
                !(F8G8 & allPieces) &&                          // squares between king and rook are empty
                !(F8G8 & threatened))                           // and not in threat from enemy pieces
            {
                // black king side castle
                if (countOnly) nMoves++;
                else addCastleMove(&nMoves, &newPositions, pos, BIT(E8), BIT(G8), BIT(H8), BIT(F8), chance);
            }
            if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&  // castle flag is set
                !(B8D8 & allPieces) &&                          // squares between king and rook are empty
                !(C8D8 & threatened))                           // and not in threat from enemy pieces
            {
                // black queen side castle
                if (countOnly) nMoves++;
                else addCastleMove(&nMoves, &newPositions, pos, BIT(E8), BIT(C8), BIT(A8), BIT(D8), chance);
            }
        }
        
        // generate king moves
#if USE_KING_LUT == 1
        uint64 kingMoves = sqKingAttacks(kingIndex);
#else
        uint64 kingMoves = kingAttacks(myKing);
#endif

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        if (countOnly)
        {
            nMoves += popCount(kingMoves);
        }
        else
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
            if (countOnly)
            {
                nMoves += popCount(knightMoves);
            }
            else
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

            if (countOnly)
            {
                nMoves += popCount(bishopMoves);
            }
            else
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

            if (countOnly)
            {
                nMoves += popCount(bishopMoves);
            }
            else
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

            if (countOnly)
            {
                nMoves += popCount(rookMoves);
            }
            else
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

            if (countOnly)
            {
                nMoves += popCount(rookMoves);
            }
            else
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
};

#if USE_TEMPLATE_CHANCE_OPT == 1
// instances of move generator
template uint32 MoveGeneratorBitboard::generateMoves<BLACK, false>(HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions);
template uint32 MoveGeneratorBitboard::generateMoves<WHITE, false>(HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions);
template uint32 MoveGeneratorBitboard::generateMoves<BLACK, true>(HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions);
template uint32 MoveGeneratorBitboard::generateMoves<WHITE, true>(HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions);
#endif



// perft counter function. Returns perft of the given board for given depth
uint64 perft_bb(HexaBitBoardPosition *pos, uint32 depth)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];

    /*
    if (depth == 2)
        printMoves = true;
    else
        printMoves = false;
    */

    uint32 nMoves = 0;
    uint8 chance = pos->chance;

    if (depth == 1)
    {
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (chance == BLACK)
        {
            nMoves = MoveGeneratorBitboard::generateMoves<BLACK, true>(pos, newPositions);
        }
        else
        {
            nMoves = MoveGeneratorBitboard::generateMoves<WHITE, true>(pos, newPositions);
        }
#else
        nMoves = MoveGeneratorBitboard::generateMoves(pos, newPositions, chance, true);
#endif
        return nMoves;
    }

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        nMoves = MoveGeneratorBitboard::generateMoves<BLACK, false>(pos, newPositions);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateMoves<WHITE, false>(pos, newPositions);
    }
#else
    nMoves = MoveGeneratorBitboard::generateMoves(pos, newPositions, chance, false);
#endif

    uint64 count = 0;

    for (uint32 i=0; i < nMoves; i++)
    {
        uint64 childPerft = perft_bb(&newPositions[i], depth - 1);
        /*if (depth == 2)
            printf("%llu\n", childPerft);*/
        count += childPerft;
    }

    return count;
}

#if TEST_GPU_PERFT == 1

#if 0
// perft search
__global__ void perft_bb_gpu(HexaBitBoardPosition *position, uint64 *generatedMoves, int depth)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32 dataIndex = blockIdx.x * MAX_MOVES + threadIdx.x;

    HexaBitBoardPosition *pos = &(position[dataIndex]);
    uint64 *moveCounter = &(generatedMoves[dataIndex]);
    
    // single shared memory variable to exchange data between threads
    __shared__ uint32 sharedBoards, sharedPerfts;

    uint32 nMoves;

    uint8 color = pos->chance;

    if (depth == 1)
    {
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (color == BLACK)
        {
            nMoves = MoveGeneratorBitboard::generateMoves<BLACK, true>(pos, NULL);
        }
        else
        {
            nMoves = MoveGeneratorBitboard::generateMoves<WHITE, true>(pos, NULL);
        }
#else
        nMoves = MoveGeneratorBitboard::generateMoves(pos, NULL, color, true);
#endif
        *moveCounter = nMoves;

        //if (index == 0)
        {
            //printf("\n Thread at depth %d, index: %d, returns count: %d", depth, index, nMoves);
        }

        return;
    }

    HexaBitBoardPosition *allChildBoards;
    uint64 *allChildPerfts;

    // only the first thread allocates memory and launches childs
    // the pointer to allocated memory is distributed to other threads
    if (threadIdx.x == 0)
    {
        int hr;
        hr = cudaMalloc(&allChildBoards, sizeof(HexaBitBoardPosition) * MAX_MOVES * blockDim.x);
        //if (hr != 0)
        //    printf("error in malloc for childBoards at depth %d\n", depth);

        hr = cudaMalloc(&allChildPerfts, sizeof(uint64) * MAX_MOVES * blockDim.x);
        //if (hr != 0)
        //    printf("error in sedond malloc at depth %d\n", depth);

        if (blockDim.x > 31)
        {
            // first thread in the thread block writes into shared memory
            sharedBoards = (int) allChildBoards;
            sharedPerfts = (int) allChildPerfts;
        }
    }

    __syncthreads();

    if (threadIdx.x > 31 && (threadIdx.x % 32) == 0)
    {
        // first thread in the warp gets value from shared memory
        allChildBoards = (HexaBitBoardPosition *) sharedBoards;
        allChildPerfts = (uint64 *) sharedPerfts;
    }

    // other threads in the warp copy from thread 0 of warp
    allChildBoards = (HexaBitBoardPosition *)   __shfl((int)allChildBoards, 0);
    allChildPerfts = (uint64 *)                 __shfl((int)allChildPerfts, 0);


    // child boards to be generated by current thread
    HexaBitBoardPosition *childBoards = &allChildBoards[threadIdx.x * MAX_MOVES];

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        nMoves = MoveGeneratorBitboard::generateMoves<BLACK, false>(pos, childBoards);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateMoves<WHITE, false>(pos, childBoards);
    }
#else
    nMoves = MoveGeneratorBitboard::generateMoves(pos, childBoards, color, false);
#endif

    if (nMoves == 0)
    {
        *moveCounter = 0;
    }
    else
    {
        uint64 *child_perfts = &allChildPerfts[threadIdx.x * MAX_MOVES];

        // only the first thread launches the grid
        if (threadIdx.x == 0)
        {
            cudaStream_t childStream;
            cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
           
            //printf("\n depth %d, block %d, thread %d, childBoards %x, childPerfts %x, nMoves %d, blockDim %d", depth, blockIdx.x, threadIdx.x, childBoards, child_perfts, nMoves, blockDim.x);
            // TODO: nMoves is going to be different for each threadBlock!
            perft_bb_gpu<<<blockDim.x, nMoves, 8, childStream>>> (allChildBoards, allChildPerfts, depth-1);
            cudaDeviceSynchronize();

            cudaStreamDestroy(childStream);
        }

        uint64 childPerft = 0;
        for (uint32 i = 0; i < nMoves; i++)
        {
            childPerft += child_perfts[i];
        }


        *moveCounter = childPerft;
    }

    if (threadIdx.x == 0)
    {
        cudaFree(allChildPerfts);
        cudaFree(allChildBoards);
    }

}
#endif

#if 0
// perft search (this version makes use of global atomics)
__global__ void perft_bb_gpu(HexaBitBoardPosition *position, uint64 *globalPerftCounter, HexaBitBoardPosition *allChildBoards, int depth)
{
    // exctact one element of work
    HexaBitBoardPosition *pos = &(position[threadIdx.x]);

    uint32 nMoves;

    uint8 color = pos->chance;

    if (depth == 1)
    {
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (color == BLACK)
        {
            nMoves = MoveGeneratorBitboard::generateMoves<BLACK, true>(pos, NULL);
        }
        else
        {
            nMoves = MoveGeneratorBitboard::generateMoves<WHITE, true>(pos, NULL);
        }
#else
        nMoves = MoveGeneratorBitboard::generateMoves(pos, NULL, color, true);
#endif
        atomicAdd(globalPerftCounter, nMoves);
        return;
    }

    HexaBitBoardPosition *childBoards = &allChildBoards[threadIdx.x * MAX_MOVES];


#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        nMoves = MoveGeneratorBitboard::generateMoves<BLACK, false>(pos, childBoards);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::generateMoves<WHITE, false>(pos, childBoards);
    }
#else
    nMoves = MoveGeneratorBitboard::generateMoves(pos, childBoards, color, false);
#endif

    if (nMoves != 0)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        HexaBitBoardPosition *childChildBoards = NULL;
        if (depth > 2)
        {
            // allocate memory for storing child boards of all childs
            int hr;
            hr = cudaMalloc(&childChildBoards, sizeof(HexaBitBoardPosition) * MAX_MOVES * nMoves);
            //if (hr != 0)
            //    printf("error in malloc for childChildBoards at depth %d\n", depth);
        }
        perft_bb_gpu<<<1, nMoves, 0, childStream>>> (childBoards, globalPerftCounter, childChildBoards, depth-1);
        cudaStreamDestroy(childStream);

        if (depth > 2)
        {
            cudaDeviceSynchronize();
            cudaFree(childChildBoards);
        }
    }
}
#endif

#define BLOCK_SIZE 256


__device__ __forceinline__ uint32 countMoves(HexaBitBoardPosition *pos, uint8 color)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateMoves<BLACK, true>(pos, NULL);
    }
    else
    {
        return MoveGeneratorBitboard::generateMoves<WHITE, true>(pos, NULL);
    }
#else
    return MoveGeneratorBitboard::generateMoves(pos, NULL, color, true);
#endif
}

__device__ __forceinline__ uint32 generateMoves(HexaBitBoardPosition *pos, uint8 color, HexaBitBoardPosition *childBoards)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateMoves<BLACK, false>(pos, childBoards);
    }
    else
    {
        return MoveGeneratorBitboard::generateMoves<WHITE, false>(pos, childBoards);
    }
#else
    return MoveGeneratorBitboard::generateMoves(pos, childBoards, color, false);
#endif
}

#if 0
// this version works quite well (peaks at around 3.7 Billion moves per second for the good position)
__global__ void perft_bb_gpu(HexaBitBoardPosition *position, uint64 *globalPerftCounter, HexaBitBoardPosition *allChildBoards, int depth, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    HexaBitBoardPosition pos = position[index];

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ uint32 movesForThread[BLOCK_SIZE];

    uint8 color = pos.chance;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    uint32 nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

/*
    if (depth == 1 && nMoves > 0)
    {
        uint64 bb  = (pos->kings | pos->pawns | pos->knights | pos->rookQueens | pos->bishopQueens);
        uint32 bb0 = (bb & 0xFFFFFFFF);
        uint32 bb1 = (bb >> 32);
        printf("\nBlock %d, thread %d, board: %X%X, moves: %d\n", blockIdx.x, threadIdx.x, bb0, bb1, nMoves);
    }
*/
    movesForThread[threadIdx.x] = nMoves;

    __syncthreads();

    if (depth == 1)
    {
        //printf("\nBlockIdx %d, threadIdx %d, nMoves: %d\n", blockIdx.x, threadIdx.x, nMoves);
        // perform reduction to sum up perfts in the thread block

        // TODO: use reduction instead of this crap
        if (threadIdx.x == 0)
        {
            for (uint32 i=1; i < blockDim.x; i++)
                nMoves += movesForThread[i];
        }

        // the first thread of the thread block uses atomicAdd to update the global perft counter
        if (threadIdx.x == 0)
        {
            //printf("\nBlockIdx %d, threadIdx %d adding %d moves to global counter\n", blockIdx.x, threadIdx.x, nMoves);
            atomicAdd(globalPerftCounter, nMoves);
        }

        return;
    }

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards

    // TODO: replace this crap with parallel scan
    uint32 allMoves = 0;
    if (threadIdx.x == 0)
    {
        //printf ("\nscan result: \n");
        for (uint32 i=0; i<blockDim.x; i++)
        {
            uint32 x = movesForThread[i];
            movesForThread[i] = allMoves;
            allMoves += x ;
            //printf("%d ", allMoves);
        }

        //printf("\nAllmoves at depth %d is %d\n", depth, allMoves);
    }


    __syncthreads();

    HexaBitBoardPosition *childBoards = &allChildBoards[blockIdx.x * BLOCK_SIZE * MAX_MOVES] + movesForThread[threadIdx.x];

    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, childBoards);
    }
    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    if (threadIdx.x == 0 && allMoves != 0)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        HexaBitBoardPosition *childChildBoards = NULL;
        if (depth > 2)
        {
            // allocate memory for storing child boards of all childs
            int hr;
            hr = cudaMalloc(&childChildBoards, sizeof(HexaBitBoardPosition) * MAX_MOVES * allMoves);
            if (hr != 0)
                printf("error in malloc for childChildBoards at depth %d, for %d moves\n", depth, allMoves);
        }

        uint32 nBlocks = (allMoves - 1) / BLOCK_SIZE + 1;
        //printf ("\nLaunching %d blocks\n", nBlocks);
        perft_bb_gpu<<<nBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(uint32), childStream>>> (childBoards, globalPerftCounter, childChildBoards, depth-1, allMoves);

        if (depth > 2)
        {
            cudaDeviceSynchronize();
            cudaFree(childChildBoards);
        }

        cudaStreamDestroy(childStream);
    }
}
#endif

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

__device__ __forceinline__ void wrapReduce(int &x)
{
    #pragma unroll
    for(int mask = 16; mask > 0 ; mask >>= 1)
        x += __shfl_xor(x, mask);
}

union sharedMemAllocs
{
    uint32 movesForThread[BLOCK_SIZE];
    struct
    {
        HexaBitBoardPosition *allChildBoards;
    };
};

// this version avoids allocating MAX_MOVES boards (so childChildBoards is never allocated by parent)
// speed ~9.4 billion moves per second in best case
__global__ void perft_bb_gpu(HexaBitBoardPosition *position, uint64 *globalPerftCounter, int depth, int nThreads)
{
    // exctact one element of work
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

#if USE_TROVE_AOS_OPT == 1
    trove::coalesced_ptr<HexaBitBoardPosition> p(position);
    HexaBitBoardPosition pos = p[index];
#else
    HexaBitBoardPosition pos = position[index];
#endif

    // shared memory structure containing moves generated by each thread in the thread block
    __shared__ sharedMemAllocs shMem;

    uint8 color = pos.chance;

    // 1. first just count the moves (and store it in shared memory for each thread in block)
    int nMoves = 0;

    if (index < nThreads)
        nMoves = countMoves(&pos, color);

    if (depth == 1)
    {
        // on Kepler, atomics are so fast that one atomic instruction per leaf node is also fast enough (faster than full reduction)!
        // wrap-wide reduction seems a little bit faster
        wrapReduce(nMoves);

        int laneId = threadIdx.x & 0x1f;

        if (laneId == 0)
        {
            atomicAdd (globalPerftCounter, nMoves);
        }
        return;
    }

    shMem.movesForThread[threadIdx.x] = nMoves;
    __syncthreads();

    // 2. perform scan (prefix sum) to figure out starting addresses of child boards
    scan(shMem.movesForThread);
    
    // convert inclusive scan to exclusive scan
    uint32 moveListOffset = shMem.movesForThread[threadIdx.x] - nMoves;

    // first thread of the block allocates memory for childBoards for the entire thread block
    uint32 allMoves = 0;
    if (threadIdx.x == 0)
    {
        allMoves = shMem.movesForThread[blockDim.x - 1];
        if (allMoves)
        {
            int hr;
            hr = cudaMalloc(&shMem.allChildBoards, sizeof(HexaBitBoardPosition) * allMoves);
            //if (hr != 0)
            //    printf("error in malloc for childBoards at depth %d, for %d moves\n", depth, allMoves);
        }
    }

    __syncthreads();

    // other threads get value from shared memory
    // address of starting of move list for the current thread
    HexaBitBoardPosition *childBoards = shMem.allChildBoards + moveListOffset;


    // 3. generate the moves now
    if (nMoves)
    {
        generateMoves(&pos, color, childBoards);
    }

    __syncthreads();

    // 4. first thread of each thread block launches new work (for moves generated by all threads in the thread block)
    if (threadIdx.x == 0 && allMoves > 0)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);
       
        HexaBitBoardPosition *childChildBoards = NULL;

        uint32 nBlocks = (allMoves - 1) / BLOCK_SIZE + 1;
        perft_bb_gpu<<<nBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(uint32), childStream>>> (childBoards, globalPerftCounter, depth-1, allMoves);

        cudaDeviceSynchronize();
        cudaFree(childBoards);
        cudaStreamDestroy(childStream);
    }
}

#endif // #if TEST_GPU_PERFT == 1