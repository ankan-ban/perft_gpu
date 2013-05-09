#include "chess.h"
#include <intrin.h>


#define DEBUG_PRINT_MOVES 1
#if DEBUG_PRINT_MOVES == 1
    bool printMoves = false;
#endif

// intel core 2 doesn't have popcnt instruction
#define USE_POPCNT 0

// use lookup table for king moves
#define USE_KING_LUT 0

// use lookup table for knight moves
#define USE_KNIGHT_LUT 0

// use lookup table (magics) for sliding moves
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

// used for castling checks
#define F8G8      C64(0x6000000000000000)
#define C8D8      C64(0x0C00000000000000)

// used to update castle flags
#define WHITE_KING_SIDE_ROOK   C64(0x0000000000000080)
#define WHITE_QUEEN_SIDE_ROOK  C64(0x0000000000000001)
#define BLACK_KING_SIDE_ROOK   C64(0x8000000000000000)
#define BLACK_QUEEN_SIDE_ROOK  C64(0x0100000000000000)
    

#define ALLSET    C64(0xFFFFFFFFFFFFFFFF)
#define EMPTY     C64(0x0)

__forceinline uint8 popCount(uint64 x)
{
#if USE_POPCNT == 1
    return __popcnt64(x);
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
__forceinline uint8 bitScan(uint64 x)
{
#ifdef _WIN64
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

class MoveGeneratorBitboard
{
private:

    // move the bits in the bitboard one square in the required direction

    __forceinline static uint64 northOne(uint64 x)
    {
        return x << 8;
    }

    __forceinline static uint64 southOne(uint64 x)
    {
        return x >> 8;
    }

    __forceinline static uint64 eastOne(uint64 x)
    {
        return (x << 1) & (~FILEA);
    }

    __forceinline static uint64 westOne(uint64 x)
    {
        return (x >> 1) & (~FILEH);
    }

    __forceinline static uint64 northEastOne(uint64 x)
    {
        return (x << 9) & (~FILEA);
    }

    __forceinline static uint64 northWestOne(uint64 x)
    {
        return (x << 7) & (~FILEH);
    }

    __forceinline static uint64 southEastOne(uint64 x)
    {
        return (x >> 7) & (~FILEA);
    }

    __forceinline static uint64 southWestOne(uint64 x)
    {
        return (x >> 9) & (~FILEH);
    }


    // fill the board in the given direction
    // taken from http://chessprogramming.wikispaces.com/


    // gen - generator  : starting positions
    // pro - propogator : empty squares / squares not of current side

    // uses kogge-stone algorithm

    __forceinline static uint64 northFill(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen;
    }

    __forceinline static uint64 southFill(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen;
    }

    __forceinline static uint64 eastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return gen;
    }
    
    __forceinline static uint64 westFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return gen;
    }


    __forceinline static uint64 northEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return gen;
    }

    __forceinline static uint64 northWestFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return gen;
    }

    __forceinline static uint64 southEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return gen;
    }

    __forceinline static uint64 southWestFill(uint64 gen, uint64 pro)
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

    __forceinline static uint64 northAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen << 8;
    }

    __forceinline static uint64 southAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen >> 8;
    }

    __forceinline static uint64 eastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return (gen << 1) & (~FILEA);
    }
    
    __forceinline static uint64 westAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return (gen >> 1) & (~FILEH);
    }


    __forceinline static uint64 northEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return (gen << 9) & (~FILEA);
    }

    __forceinline static uint64 northWestAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return (gen << 7) & (~FILEH);
    }

    __forceinline static uint64 southEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return (gen >> 7) & (~FILEA);
    }

    __forceinline static uint64 southWestAttacks(uint64 gen, uint64 pro)
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

    __forceinline static uint64 bishopAttacks(uint64 bishops, uint64 pro)
    {
        return northEastAttacks(bishops, pro) |
               northWestAttacks(bishops, pro) |
               southEastAttacks(bishops, pro) |
               southWestAttacks(bishops, pro) ;
    }

    __forceinline static uint64 rookAttacks(uint64 rooks, uint64 pro)
    {
        return northAttacks(rooks, pro) |
               southAttacks(rooks, pro) |
               eastAttacks (rooks, pro) |
               westAttacks (rooks, pro) ;
    }

    __forceinline static uint64 queenAttacks(uint64 queens, uint64 pro)
    {
        return rookAttacks  (queens, pro) |
               bishopAttacks(queens, pro) ;
    }

    __forceinline static uint64 kingAttacks(uint64 kingSet) 
    {
        uint64 attacks = eastOne(kingSet) | westOne(kingSet);
        kingSet       |= attacks;
        attacks       |= northOne(kingSet) | southOne(kingSet);
        return attacks;
    }

    // efficient knight attack generator
    // http://chessprogramming.wikispaces.com/Knight+Pattern
    __forceinline static uint64 knightAttacks(uint64 knights) {
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
    __forceinline static uint64 getOne(uint64 x)
    {
        return x & (-x);
    }

    __forceinline static bool isMultiple(uint64 x)
    {
        return x ^ getOne(x);
    }

    __forceinline static bool isSingular(uint64 x)
    {
        return !isMultiple(x); 
    }


public:

    // finds the squares in between the two given squares
    // taken from 
    // http://chessprogramming.wikispaces.com/Square+Attacked+By#Legality Test-In Between-Pure Calculation
    // Ankan : TODO: this doesn't seem to work for G8 - B3
    __forceinline static uint64 squaresInBetween(uint8 sq1, uint8 sq2)
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
    __forceinline static uint64 squaresInLine(uint8 sq1, uint8 sq2)
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
    }


    static uint64 findPinnedPieces (uint64 myKing, uint64 myPieces, uint64 enemyBishops, uint64 enemyRooks, uint64 allPieces, uint8 kingIndex)
    {
        // check for sliding attacks to the king's square

        // It doesn't matter if we process more attackers behind the first attackers
        // They will be taken care of when we check for no. of obstructing squares between king and the attacker
        /*
        uint64 b = bishopAttacks(myKing, ~enemyPieces) & enemyBishops;
        uint64 r = rookAttacks  (myKing, ~enemyPieces) & enemyRooks;
        */
        uint64 b = BishopAttacks[kingIndex] & enemyBishops;
        uint64 r = RookAttacks  [kingIndex] & enemyRooks;
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

            uint64 squaresInBetween = Between[attackerIndex][kingIndex]; // same as using obstructed() function
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
    static uint64 findAttackedSquares(uint64 emptySquares, uint64 enemyBishops, uint64 enemyRooks, 
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
    __forceinline static void addMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *newBoard)
    {
        **newPos = *newBoard;
        (*newPos)++;
        (*nMoves)++;
    }


    __forceinline static void updateCastleFlag(HexaBitBoardPosition *pos, uint64 dst)
    {
        // TODO: might want to try some bitwise operator magic 
        //       or if/else on chance (don't forget to modify addSlidingMove if you decide to add if/else on chance)
        if (dst & BLACK_KING_SIDE_ROOK)
            pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & BLACK_QUEEN_SIDE_ROOK)
            pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;

        if (dst & WHITE_KING_SIDE_ROOK)
            pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & WHITE_QUEEN_SIDE_ROOK)
            pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
    }

    __forceinline static void addSlidingMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
        updateCastleFlag(&newBoard, src | dst);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    __forceinline static void addKnightMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
        updateCastleFlag(&newBoard, dst);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    __forceinline static void addKingMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
        updateCastleFlag(&newBoard, dst);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }


    __forceinline static void addCastleMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
    __forceinline static void addSinglePawnMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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

    static void addEnPassentMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
    __forceinline static void addPawnMoves(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
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
            updateCastleFlag(&newBoard, dst);

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


    static uint32 generateMovesOutOfCheck (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened, 
                                           uint8 chance, uint8 kingIndex)
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
        uint64 kingMoves = KingAttacks[kingIndex];
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
            uint64 safeSquares = attackers | Between[kingIndex][bitScan(attackers)];
            
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
                if (dst) addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);

                // en-passent 
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst) addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (pos->knights & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
#if USE_KNIGHT_LUT == 1
                uint64 knightMoves = KnightAttacks[bitScan(knight)] & safeSquares;
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


    // TODO: have another function called countMoves... that only counts the no. of valid moves without generating them
    // or maybe make it tempelated function ?

    // generates moves for the given board position
    // returns the no of moves generated
    // newPositions contains the new positions after making the generated moves
    // returns only count if newPositions is NULL
    static uint32 generateMoves (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions)
    {
        uint32 nMoves = 0;

        uint8 chance = pos->chance;

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
            return generateMovesOutOfCheck(pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, 
                                           pinned, threatened, chance, kingIndex);
        }

        // 1. pawn moves
        uint64 myPawns = allPawns & myPieces;

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = Line[pawnIndex][kingIndex];

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
            if (dst) 
            {
                addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, false, pawnIndex);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) addSinglePawnMove(&nMoves, &newPositions, pos, pawn, dst, chance, true, pawnIndex);
            }

            line &= enemyPieces;

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            if (dst) addPawnMoves(&nMoves, &newPositions, pos, pawn, dst, chance);

            // en-passent capture isn't possible by a pinned pawn
            // TODO: think more about it


            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

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
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) 
            {
                // if the enPassent captured piece, the pawn and the king all lie in the same line, 
                // we need to check if the enpassent would move the king into check!!
                // really painful condition!!@!
                uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                bool causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & 
                                   (pos->kings & myPieces);
                if (!causesCheck)
                {
                    addEnPassentMove(&nMoves, &newPositions, pos, pawn, dst, chance);
                }
            }


            myPawns ^= pawn;
        }

        // generate castling moves
        // TODO: maybe it's better not to branch on chance/color (or maybe it is once we move to templated generateMoves on color)
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
                !(C1D1 & allPieces) &&                          // squares between king and rook are empty
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
                !(C8D8 & allPieces) &&                          // squares between king and rook are empty
                !(C8D8 & threatened))                           // and not in threat from enemy pieces
            {
                // black queen side castle
                addCastleMove(&nMoves, &newPositions, pos, BIT(E8), BIT(C8), BIT(A8), BIT(D8), chance);
            }
        }
        
        // generate king moves
#if USE_KING_LUT == 1
        uint64 kingMoves = KingAttacks[kingIndex];
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
            uint64 knightMoves = KnightAttacks[bitScan(knight)] & ~myPieces;
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
            bishopMoves &= Line[bitScan(bishop)][kingIndex];    // pined sliding pieces can move only along the line
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
            rookMoves &= Line[bitScan(rook)][kingIndex];    // pined sliding pieces can move only along the line
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
};
