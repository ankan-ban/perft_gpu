#ifndef CHESS_H
#define CHESS_H

#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <assert.h>

#define S_OK 0

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


// bits 012 - piece
// bit    4 - color
/*
#define COLOR_PIECE(color, piece)			((color << 4) | (piece))
#define COLOR(colorpiece)					((colorpiece) >> 4)
#define PIECE(colorpiece)					((colorpiece) & 7)
#define EMPTY_SQUARE						0
#define ISEMPTY(colorpiece)					((colorpiece) == EMPTY_SQUARE)
#define IS_OF_COLOR(colorpiece, color)		(colorpiece && (COLOR(colorpiece) == color))
#define IS_ENEMY_COLOR(colorpiece, color)	(colorpiece && (COLOR(colorpiece) != color))
*/

// new encoding for fast table based move generation
// bits 01234 : color
// bits   567 : piece

/* From http://chessprogramming.wikispaces.com/Table-driven+Move+Generation
 five least significant piece code bits from board: 
   01000 empty
   10101 white piece (0x15)
   10110 black piece (0x16)*/
/*
#define COLOR_PIECE(color, piece)      		((0x15 + color) | (piece << 5))
#define COLOR(colorpiece)              		(((colorpiece & 2) >> 1))
#define PIECE(colorpiece)              		((colorpiece) >> 5)
#define EMPTY_SQUARE						0x8
#define ISEMPTY(colorpiece)					((colorpiece) & EMPTY_SQUARE)
#define IS_OF_COLOR(colorpiece, color)		((colorpiece) & (1 << (color)))
#define IS_ENEMY_COLOR(colorpiece, color)	(IS_OF_COLOR(colorpiece, 1 - color))
*/

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

/*
       The board representation     Free space for other structures

	A	B	C	D	E	F	G	H
8	70	71	72	73	74	75	76	77	78	79	7A	7B	7C	7D	7E	7F
7	60	61	62	63	64	65	66	67	68	69	6A	6B	6C	6D	6E	6F
6	50	51	52	53	54	55	56	57	58	59	5A	5B	5C	5D	5E	5F
5	40	41	42	43	44	45	46	47	48	49	4A	4B	4C	4D	4E	4F
4	30	31	32	33	34	35	36	37	38	39	3A	3B	3C	3D	3E	3F
3	20	21	22	23	24	25	26	27	28	29	2A	2B	2C	2D	2E	2F
2	10	11	12	13	14	15	16	17	18	19	1A	1B	1C	1D	1E	1F
1	00	01	02	03	04	05	06	07	08	09	0A	0B	0C	0D	0E	0F

*/


// size 4 bytes
struct Move
{
    uint8  src;             // source position of the piece
    uint8  dst;             // destination position
    uint8  capturedPiece;   // the piece captured (if any)
    uint8  flags;           // flags to indicate special moves, e.g castling, en passent, promotion, etc
};
CT_ASSERT(sizeof(Move) == 4);


// position of the board in bit-board representation
struct QuadBitBoardPosition
{
    // 32 bytes of dense bitboard data
    uint64   black;   // 1 - black, 0 - white/empty
    uint64   PBQ;     // pawns, bishops and queens
    uint64   NB;      // knights and bishops
    uint64   RQK;     // rooks, queens and kings

    // 8 bytes of state / free space
    uint8    chance;            // whose move it is
    uint8    whiteCastle;       // whether white can castle
    uint8    blackCastle;       // whether black can castle
    uint8    enPassent;         // col + 1 (where col is the file on which enpassent is possible)
    uint8    halfMoveCounter;   // to detect 50 move draw rule
    uint8    padding[3];        // free space to store additional info if needed
};
CT_ASSERT(sizeof(QuadBitBoardPosition) == 40);

// another bit-board based board representation using 6 bitboards
struct HexaBitBoardPosition
{
    // 48 bytes of bitboard data with interleaved game state data in pawn bitboards
    uint64   whitePieces;
    union
    {
        uint64   pawns;
        struct 
        {
            uint8 whiteCastle       : 2;
            uint8 blackCastle       : 2;
            uint8 enPassent         : 4;         // file + 1 (file is the file containing the enpassent-target pawn)
            uint8 padding[6];
            uint8 halfMoveCounter   : 7;         // to detect 50 move draw rule
            uint8 chance            : 1;
        };
    };
    uint64   knights;
    uint64   bishopQueens;
    uint64   rookQueens;
    uint64   kings;
};
CT_ASSERT(sizeof(HexaBitBoardPosition) == 48);

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

#if 0
    CUDA_CALLABLE_MEMBER bool operator == (CMove a) const {return (m_Move == a.m_Move);}
    CUDA_CALLABLE_MEMBER bool operator != (CMove a) const {return (m_Move != a.m_Move);}

    CUDA_CALLABLE_MEMBER void operator = (CMove a) 
    {
        m_Move = a.m_Move;
    }
#endif 
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

// might want to use these flags:
/*
code	promotion	capture	special 1	special 0	kind of move
0	    0        	0    	0        	0        	quiet moves
1	    0        	0    	0        	1        	double pawn push
2	    0        	0    	1        	0        	king castle
3	    0        	0    	1        	1        	queen castle
4	    0        	1    	0        	0        	captures
5	    0        	1    	0        	1        	ep-capture
8	    1        	0    	0        	0        	knight-promotion
9	    1        	0    	0        	1        	bishop-promotion
10	    1        	0    	1        	0        	rook-promotion
11	    1        	0    	1        	1        	queen-promotion
12	    1        	1    	0        	0        	knight-promo capture
13	    1        	1    	0        	1        	bishop-promo capture
14	    1        	1    	1        	0        	rook-promo capture
15	    1        	1    	1        	1        	queen-promo capture
*/


// max no of moves possible for a given board position (this can be as large as 218 ?)
// e.g, test this FEN string "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"
#define MAX_MOVES 256

#define MAX_GAME_LENGTH 300

// max no of moves possible by a single piece
// actually it's 27 for a queen when it's in the center of the board
#define MAX_SINGLE_PIECE_MOVES 32

// random numbers for zobrist hashing
struct ZobristRandoms
{
    uint64 pieces[2][6][64];     // position of every piece on board
    uint64 castlingRights[2][2]; // king side and queen side castle for each side
    uint64 enPassentTarget[8];   // 8 possible files for en-passent target (if any)
    uint64 chance;               // chance (side to move)
    uint64 depth;                // search depth (used only by perft)
};


// indexes used to reference the zobristRandoms.pieces[] table above
#define ZOB_INDEX_PAWN     (PAWN - 1  )
#define ZOB_INDEX_KNIGHT   (KNIGHT - 1)
#define ZOB_INDEX_BISHOP   (BISHOP - 1)
#define ZOB_INDEX_ROOK     (ROOK - 1  )
#define ZOB_INDEX_QUEEN    (QUEEN - 1 )
#define ZOB_INDEX_KING     (KING - 1  )
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

// hash table entry for Perft
struct HashEntryPerft
{
    union
    {
        struct 
        {
            union
            {
                uint64 hashKey;
                struct
                {
                    // 8 LSB's are not important as the hash table size is at least > 256 entries
                    // store depth in the 8 LSB's
                    uint8 depth;
                    uint8 hashPart[7];  // most significant bits of the hash key
                };
            };
            uint64 perftVal;
        };
#ifdef __CUDA_ARCH__
        uint4 rawVal;
#endif
    };
};

CT_ASSERT(sizeof(HashEntryPerft) == 16);

// Paul B's method of storing two entries per slot of hash table
struct DualHashEntry
{
    HashEntryPerft deepest;
    HashEntryPerft mostRecent;
};
CT_ASSERT(sizeof(DualHashEntry) == 32);

struct ShallowHashEntry
{
    union
    {
        uint64 hashKey;     // most significant 40 bits used

        // 24 LSB's are not important as the hash table size is at least 2 ^ 24 entries
        // store perft (only for shallow depths) in the 24 LSB's
        uint64 perftVal;    // least significant 24 bits used
    };
};
CT_ASSERT(sizeof(ShallowHashEntry) == 8);

// 128-bit hash keys for deep-perfts:
union HashKey128b
{
    struct
    {
        uint64 lowPart;
        uint64 highPart;
    };
#ifdef __CUDA_ARCH__
    uint4 rawVal;
#endif
    CUDA_CALLABLE_MEMBER HashKey128b() { lowPart = highPart = 0ull; }
    CUDA_CALLABLE_MEMBER HashKey128b(uint64 l, uint64 h) { lowPart = l, highPart = h; }
    
    CUDA_CALLABLE_MEMBER HashKey128b operator^(const HashKey128b& b)
    {
        HashKey128b temp;
        temp.lowPart = this->lowPart ^ b.lowPart;
        temp.highPart = this->highPart ^ b.highPart;
        return temp;
    }

    CUDA_CALLABLE_MEMBER HashKey128b operator^=(const HashKey128b& b)
    {
        this->lowPart = this->lowPart ^ b.lowPart;
        this->highPart = this->highPart ^ b.highPart;
        return *this;
    }
};
CT_ASSERT(sizeof(HashKey128b) == 16);

struct ShallowHashEntry128b
{
    union
    {
        struct {
            HashKey128b hashKey;     // most significant 40 + 64 bits used
        };
        struct
        {
            // 24 LSB's are not important as the hash table size is at least 2 ^ 24 entries
            // store perft (only for shallow depths) in the 24 LSB's
            uint64 perftVal;    // least significant 24 bits used (aliased with hashKey.lowPart)
            uint64 padding;
        };
    };
};
CT_ASSERT(sizeof(ShallowHashEntry128b) == 16);

// hash table entry for Perft
#pragma pack(push, 1)
struct HashEntryPerft128b
{
    // WAR for silly compiler bug??
    // it says that it can't call default constructor because it was deleted!!@??
#if __CUDA_ARCH__
   __device__ __host__ 
#endif
    HashEntryPerft128b() { }

    union
    {
        struct
        {
            HashKey128b hashKey;
        };
        struct
        {
            // 8 LSB's are not important as the hash table size is at least > 256 entries
            // store depth in the 8 LSB's
            uint8 depth;
            uint8 hashPart[15];  // most significant bits of the hash key
        };
    };
    uint64 perftVal;
};
#pragma pack(pop)

// WHY THIS CT_ASSERT FAILS.. even when size if 24 ???
// it's sometimes 24 andd sometimes 32 -> very bad compiler, fixed with setting #pragma pack(push, 1) - which is good!
// CT_ASSERT((sizeof(HashEntryPerft128b) == 24) || (sizeof(HashEntryPerft128b) == 32));
// TODO: check if it can cause problem with 128 bit atomic read/writes?
CT_ASSERT(sizeof(HashEntryPerft128b) == 24);

#include "utils.h"


struct CompleteHashEntry
{
    uint64 perft;       // perft value
    uint64 hashLow;     // low part of 128 bit hash
    uint32 hashHigh;    // 32 LSBs of highPart of 128b hash
    uint32 next;
};
CT_ASSERT(sizeof(CompleteHashEntry) == 24);

// most compact board representation using huffman coding (doesn't work for all positions)
// those positions are stored in a different place
struct CompactPosRecord
{
    /************************************
    Empty square = 0                         32
    Pawn         = 10c               3*16    48 
    Bishop       = 1100c             5*4     20
    Knight       = 1101c             5*4     20
    Rook         = 1110c             5*4     20
    Queen        = 11110c            6*2     12
    King         = 11111c            6*2     12

    164 bits->entire board position (at least start pos)!
        --- huffman coded!
    9 bit of state :
        color to move     (1 bit)
        castling rights   (4 bits)
        en-passant column (4 bits)
        ==================================
        For total of 173 bits...
    30 bit index of next entry (=> table can't be more than of 1 billion elements)
    53 bit of perft (let's hope this is sufficient?)
    256 byte entry
    **************************************/
    union {
        uint64 bf[4];
        struct
        {
            uint64 perftVal       : 53;
            uint64 nextLow        : 11;

            uint64 nextHigh       : 19;
            uint64 chance         :  1;
            uint64 whiteCastle    :  2;
            uint64 blackCastle    :  2;
            uint64 enPassent      :  4;         // file + 1 (file is the file containing the enpassent-target pawn)
            uint64 compressedPos0 : 36;

            uint64 compressedPos1;

            uint64 compressedPos2;
        };
    };

    // write the bit at the given index into the compressed pos part
    void writeBit(int index, int val)
    {
        if (index < 36)
        {
            if (val)
                compressedPos0 |= BIT(index);
            else
                compressedPos0 &= ~(BIT(index));
        }
        else if (index < 36 + 64)
        {
            index = index - 36;
            if (val)
                compressedPos1 |= BIT(index);
            else
                compressedPos1 &= ~(BIT(index));
        }
        else
        {
            index = index - 36 - 64;
            if (val)
                compressedPos2 |= BIT(index);
            else
                compressedPos2 &= ~(BIT(index));
        }
    }

    int getBit(int index)
    {
        if (index < 36)
        {
            return !!(compressedPos0 & BIT(index));
        }
        else if (index < 36 + 64)
        {
            index = index - 36;
            return !!(compressedPos1 & BIT(index));
        }
        else
        {
            index = index - 36 - 64;
            return !!(compressedPos2 & BIT(index));
        }
    }

    bool encodePos(HexaBitBoardPosition *pos, uint64 val, uint32 nextIndex)
    {
        chance      = pos->chance;
        whiteCastle = pos->whiteCastle;
        blackCastle = pos->blackCastle;
        enPassent   = pos->enPassent;
        perftVal    = val;
        nextLow     = nextIndex & 0x7FF;
        nextHigh    = (nextIndex >> 11) & 0x7FFFF;

        // huffman encode the position
        BoardPosition pos88;
        Utils::boardHexBBTo088(&pos88, pos);

        int i, j;
        int index088 = 0;
        int bitIndex = 0;

        for (i = 7; i >= 0; i--)
        {
            for (j = 0; j < 8; j++)
            {
                uint8 code = pos88.board[index088];
                uint8 c     = COLOR(code);
                uint8 piece = PIECE(code);
                switch (piece)
                {
                case EMPTY_SQUARE:
                    writeBit(bitIndex++, 0);
                    break;
                case PAWN:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, c);
                    break;
                case KNIGHT:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, c);
                    break;
                case BISHOP:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, c);
                    break;
                case ROOK:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, c);
                    break;
                case QUEEN:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 0);
                    writeBit(bitIndex++, c);
                    break;
                case KING:
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, 1);
                    writeBit(bitIndex++, c);
                    break;
                }
                index088++;
            }
            // skip 8 cells of padding
            index088 += 8;
        }

        // overflow!
        if (bitIndex > 164)
        {
            memset(this, 0, sizeof(this));
            return false;
        }

        return true;
    }

    bool decodePos(HexaBitBoardPosition *pos, uint64 *val, uint32 *nextIndex)
    {
        // huffman decode position:
        int i, j;
        int index088 = 0;
        int bitIndex = 0;
        BoardPosition pos88;

        for (i = 7; i >= 0; i--)
        {
            for (j = 0; j < 8; j++)
            {
                uint8 code;
                uint8 color;
                uint8 piece;

                int bit = getBit(bitIndex++);
                if (bit == 0)
                {   // 0
                    code = EMPTY_SQUARE;
                }
                else
                {   // 1
                    bit = getBit(bitIndex++);
                    if (bit == 0)
                    {   // 10c -> PAWN
                        piece = PAWN;
                        color = getBit(bitIndex++);
                        code = COLOR_PIECE(color, piece);
                    }
                    else
                    {
                        // 11
                        bit = getBit(bitIndex++);
                        if (bit == 0)
                        {
                            // 110 -> knight/bishop
                            bit = getBit(bitIndex++);
                            if (bit == 0)
                            {
                                // 1100c -> bishop
                                piece = BISHOP;
                                color = getBit(bitIndex++);
                                code = COLOR_PIECE(color, piece);
                            }
                            else
                            {
                                // 1101c -> knight
                                piece = KNIGHT;
                                color = getBit(bitIndex++);
                                code = COLOR_PIECE(color, piece);
                            }
                        }
                        else
                        {   // 111
                            bit = getBit(bitIndex++);
                            if (bit == 0)
                            {
                                // 1110c -> rook
                                piece = ROOK;
                                color = getBit(bitIndex++);
                                code = COLOR_PIECE(color, piece);
                            }
                            else
                            {
                                // 1111
                                bit = getBit(bitIndex++);
                                if (bit == 0)
                                {
                                    // 11110c -> Queen
                                    piece = QUEEN;
                                    color = getBit(bitIndex++);
                                    code = COLOR_PIECE(color, piece);
                                }
                                else
                                {
                                    // 11111c -> King
                                    piece = KING;
                                    color = getBit(bitIndex++);
                                    code = COLOR_PIECE(color, piece);
                                }
                            }
                        }
                    }
                }

                pos88.board[index088] = code;

                index088++;
            }
            // skip 8 cells of padding
            index088 += 8;
        }

        Utils::board088ToHexBB(pos, &pos88);

        pos->chance = chance;
        pos->whiteCastle = whiteCastle;
        pos->blackCastle = blackCastle;
        pos->enPassent = enPassent;
        *val = perftVal;
        *nextIndex = (nextHigh << 11) | nextLow;
    }
};

CT_ASSERT(sizeof(CompactPosRecord) == 32);



#endif