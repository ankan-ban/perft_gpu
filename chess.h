#ifndef CHESS_H
#define CHESS_H

#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <assert.h>

#ifndef S_OK
#define S_OK 0
#endif

// various compile time settings
#include "switches.h"

typedef unsigned char      uint8;
typedef unsigned short     uint16;
typedef unsigned int       uint32;
typedef unsigned long long uint64;

#define HI(x) ((uint32)((x)>>32))
#define LO(x) ((uint32)(x))

#define CT_ASSERT(expr) \
int __static_assert(int static_assert_failed[(expr)?1:-1])

#define BIT(i)   (1ULL << (i))

// Terminology:
//
// file - column [A - H]
// rank - row    [1 - 8]


// piece constants
#define PAWN    1
#define KNIGHT  2
#define BISHOP  3
#define ROOK    4
#define QUEEN   5
#define KING    6

// chance (side) constants
#define WHITE   0
#define BLACK   1

#define SLIDING_PIECE_INDEX(piece) ((piece) - BISHOP)
// BISHOP 0
// ROOK   1
// QUEEN  3


// another encoding (a bit faster and simpler than above)
// bits  01 color	1 - white, 2 - black
// bits 234 piece
#define COLOR_PIECE(color, piece)      		((1+color) | (piece << 2))
#define COLOR(colorpiece)              		(((colorpiece & 2) >> 1))
#define PIECE(colorpiece)              		((colorpiece) >> 2)
#define EMPTY_SQUARE						0x0
#define ISEMPTY(colorpiece)					(!(colorpiece))
#define IS_OF_COLOR(colorpiece, color)		((colorpiece) & (1 << (color)))
#define IS_ENEMY_COLOR(colorpiece, color)	(IS_OF_COLOR(colorpiece, 1 - color))



#define INDEX088(rank, file)        ((rank) << 4 | (file))
#define RANK(index088)              (index088 >> 4)
#define FILE(index088)              (index088 & 0xF)

#define ISVALIDPOS(index088)        (((index088) & 0x88) == 0)

// special move flags
#define CASTLE_QUEEN_SIDE  1
#define CASTLE_KING_SIDE   2
#define EN_PASSENT         3
#define PROMOTION_QUEEN    4
#define PROMOTION_ROOK     5
#define PROMOTION_BISHOP   6
#define PROMOTION_KNIGHT   7

// castle flags in board position (1 and 2)
#define CASTLE_FLAG_KING_SIDE   1
#define CASTLE_FLAG_QUEEN_SIDE  2


enum eSquare
{
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};

// size 128 bytes
// let's hope this fits in register file
struct BoardPosition
{
    union
    {
        uint8 board[128];    // the 0x88 board

        struct
        {
            uint8 row0[8];  uint8 padding0[3];

            uint8 chance;           // whose move it is
            uint8 whiteCastle;      // whether white can castle
            uint8 blackCastle;      // whether black can castle
            uint8 enPassent;        // col + 1 (where col is the file on which enpassent is possible)
            uint8 halfMoveCounter;  // to detect draw using 50 move rule

            uint8 row1[8]; uint8 padding1[8];
            uint8 row2[8]; uint8 padding2[8];
            uint8 row3[8]; uint8 padding3[8];
            uint8 row4[8]; uint8 padding4[8];
            uint8 row5[8]; uint8 padding5[8];
            uint8 row6[8]; uint8 padding6[8];
            uint8 row7[8]; uint8 padding7[8];

            // 60 unused bytes (padding) available for storing other structures if needed
        };
    };
};

CT_ASSERT(sizeof(BoardPosition) == 128);

// size 4 bytes
struct Move
{
    uint8  src;             // source position of the piece
    uint8  dst;             // destination position
    uint8  capturedPiece;   // the piece captured (if any)
    uint8  flags;           // flags to indicate special moves, e.g castling, en passent, promotion, etc
};
CT_ASSERT(sizeof(Move) == 4);


// Quad-bitboard representation: 4 bitboards encode piece type + color per square
// bb[0] = color bit (1 for black pieces, 0 for white/empty)
// bb[1] = piece bit 0 (set for PAWN, BISHOP, QUEEN)
// bb[2] = piece bit 1 (set for KNIGHT, BISHOP, KING)
// bb[3] = piece bit 2 (set for ROOK, QUEEN, KING)
// Piece encoding: PAWN=001, KNIGHT=010, BISHOP=011, ROOK=100, QUEEN=101, KING=110
struct QuadBitBoard
{
    uint64 bb[4];
};
CT_ASSERT(sizeof(QuadBitBoard) == 32);

// Game state stored separately for better coalescing
struct GameState
{
    uint16 whiteCastle      : 2;
    uint16 blackCastle      : 2;
    uint16 enPassent        : 4;   // file + 1
    uint16 halfMoveCounter  : 7;
    uint16 chance           : 1;   // 0=WHITE, 1=BLACK
};
CT_ASSERT(sizeof(GameState) == 2);

// a more compact move structure (16 bit)
// from http://chessprogramming.wikispaces.com/Encoding+Moves
class CMove
{
public:

    CUDA_CALLABLE_MEMBER CMove(uint8 from, uint8 to, uint8 flags)
    {
        m_Move = ((flags & 0xF) << 12) | ((to & 0x3F) << 6) | (from & 0x3F);
    }

    CUDA_CALLABLE_MEMBER CMove()
    {
        m_Move = 0;
    }

    CUDA_CALLABLE_MEMBER unsigned int getTo()    const {return (m_Move >> 6)  & 0x3F;}
    CUDA_CALLABLE_MEMBER unsigned int getFrom()  const {return (m_Move)       & 0x3F;}
    CUDA_CALLABLE_MEMBER unsigned int getFlags() const {return (m_Move >> 12) & 0x0F;}

protected:

   uint16 m_Move;

};

CT_ASSERT(sizeof(CMove) == 2);

enum eCompactMoveFlag
{
    CM_FLAG_QUIET_MOVE        = 0,

    CM_FLAG_DOUBLE_PAWN_PUSH  = 1,

    CM_FLAG_KING_CASTLE       = 2,
    CM_FLAG_QUEEN_CASTLE      = 3,

    CM_FLAG_CAPTURE           = 4,
    CM_FLAG_EP_CAPTURE        = 5,


    CM_FLAG_PROMOTION         = 8,

    CM_FLAG_KNIGHT_PROMOTION  = 8,
    CM_FLAG_BISHOP_PROMOTION  = 9,
    CM_FLAG_ROOK_PROMOTION    = 10,
    CM_FLAG_QUEEN_PROMOTION   = 11,

    CM_FLAG_KNIGHT_PROMO_CAP  = 12,
    CM_FLAG_BISHOP_PROMO_CAP  = 13,
    CM_FLAG_ROOK_PROMO_CAP    = 14,
    CM_FLAG_QUEEN_PROMO_CAP   = 15,
};

struct FancyMagicEntry
{
    union
    {
        struct {
            uint64  factor;     // the magic factor
            int     position;   // position in the main lookup table (of 97264 entries)

            int     offset;     // position in the byte lookup table (only used when byte lookup is enabled)
        };
#ifndef SKIP_CUDA_CODE
#ifdef __CUDA_ARCH__
        uint4 data;
#endif
#endif
    };
};

// max no of moves possible for a given board position (this can be as large as 218 ?)
// e.g, test this FEN string "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"
#define MAX_MOVES 256

#define MAX_GAME_LENGTH 300

// max no of moves possible by a single piece
// actually it's 27 for a queen when it's in the center of the board
#define MAX_SINGLE_PIECE_MOVES 32

#include "utils.h"


#endif
