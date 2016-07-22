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
    struct
    {
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
};
#pragma pack(pop)

// WHY THIS CT_ASSERT FAILS.. even when size if 24 ???
// it's sometimes 24 andd sometimes 32 -> very bad compiler, fixed with setting #pragma pack(push, 1) - which is good!
// CT_ASSERT((sizeof(HashEntryPerft128b) == 24) || (sizeof(HashEntryPerft128b) == 32));
// TODO: check if it can cause problem with 128 bit atomic read/writes?
CT_ASSERT(sizeof(HashEntryPerft128b) == 24);

/** Declarations for class/methods in Util.cpp **/

// utility functions for reading FEN String, EPD file, displaying board, etc

/*
   Three kinds of board representations are used with varying degrees of readibility and efficiency

   1. Human readable board (EPD file?)
 
	  e.g. for Starting Position:

	  rnbqkbnr
	  pppppppp
	  ........
	  ........
	  ........
	  ........
	  PPPPPPPP
	  RNBQKBNR

  2. FEN string board e.g: 
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  3. 0x88 binary board described at the begining of this file

  The first two fromats are only used for displaying/taking input and interacting with the UCI protocol
  For all other purposes the 0x88 board is used

*/



// used by the getPieceChar function
class Utils {

private:

	// gets the numeric code of the piece represented by a character
	static uint8 getPieceCode(char piece); 

	// Gets char representation of a piece code
	//static char getPieceChar(uint8 code);

    CUDA_CALLABLE_MEMBER static char getPieceChar(uint8 code)
    {
        const char pieceCharMapping[] = { '.', 'P', 'N', 'B', 'R', 'Q', 'K' };

        uint8 color = COLOR(code);
        uint8 piece = PIECE(code);
        char pieceChar = '.';

        if (code != EMPTY_SQUARE)
        {
            assert((color == WHITE) || (color == BLACK));
            assert((piece >= PAWN) && (piece <= KING));
            pieceChar = pieceCharMapping[piece];
        }

        if (color == BLACK)
        {
            pieceChar += ('p' - 'P');
        }
        return pieceChar;
    }


public:	

	// reads a board from text file
	static void readBoardFromFile(char filename[], char board[8][8]); 
    static void readBoardFromFile(char filename[], BoardPosition *pos);


	// displays the board in human readable form
	//static void dispBoard(char board[8][8]); 
    //static void dispBoard(BoardPosition *pos);

    // displays a move in human readable form
    static void displayMove(Move move);

    // methods to display the board (in the above form)

    CUDA_CALLABLE_MEMBER static void dispBoard(char board[8][8])
    {
        int i, j;
        for (i = 0; i<8; i++) {
            for (j = 0; j<8; j++)
                printf("%c", board[i][j]);
            printf("\n");
        }
    }

    // convert to char board
    CUDA_CALLABLE_MEMBER static void board088ToChar(char board[8][8], BoardPosition *pos)
    {
        int i, j;
        int index088 = 0;

        for (i = 7; i >= 0; i--)
        {
            for (j = 0; j < 8; j++)
            {
                char piece = getPieceChar(pos->board[index088]);
                board[i][j] = piece;
                index088++;
            }
            // skip 8 cells of padding
            index088 += 8;
        }
    }


    // convert bitboard to 088 board
    CUDA_CALLABLE_MEMBER static void boardHexBBTo088(BoardPosition *pos088, HexaBitBoardPosition *posBB)
    {
        memset(pos088, 0, sizeof(BoardPosition));


        uint64 queens = posBB->bishopQueens & posBB->rookQueens;

#define RANKS2TO7 0x00FFFFFFFFFFFF00ull
        uint64 pawns = posBB->pawns & RANKS2TO7;

        uint64 allPieces = posBB->kings | posBB->knights | pawns | posBB->rookQueens | posBB->bishopQueens;

        for (uint8 i = 0; i<64; i++)
        {
            uint8 rank = i >> 3;
            uint8 file = i & 7;
            uint8 index088 = INDEX088(rank, file);

            if (allPieces & BIT(i))
            {
                uint8 color = (posBB->whitePieces & BIT(i)) ? WHITE : BLACK;
                uint8 piece = 0;
                if (posBB->kings & BIT(i))
                {
                    piece = KING;
                }
                else if (posBB->knights & BIT(i))
                {
                    piece = KNIGHT;
                }
                else if (pawns & BIT(i))
                {
                    piece = PAWN;
                }
                else if (queens & BIT(i))
                {
                    piece = QUEEN;
                }
                else if (posBB->bishopQueens & BIT(i))
                {
                    piece = BISHOP;
                }
                else if (posBB->rookQueens & BIT(i))
                {
                    piece = ROOK;
                }
                assert(piece);

                pos088->board[index088] = COLOR_PIECE(color, piece);
            }
        }

        pos088->chance = posBB->chance;
        pos088->blackCastle = posBB->blackCastle;
        pos088->whiteCastle = posBB->whiteCastle;
        pos088->enPassent = posBB->enPassent;
        pos088->halfMoveCounter = posBB->halfMoveCounter;
    }


    CUDA_CALLABLE_MEMBER static void dispBoard(BoardPosition *pos)
    {
        char board[8][8];
        board088ToChar(board, pos);

        printf("\nBoard Position: \n");
        dispBoard(board);
        printf("\nGame State: \n");

        if (pos->chance == WHITE)
        {
            printf("White to move\n");
        }
        else
        {
            printf("Black to move\n");
        }

        if (pos->enPassent)
        {
            printf("En passent allowed for file: %d\n", pos->enPassent);
        }

        printf("Allowed Castlings:\n");

        if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
            printf("White King Side castle\n");

        if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
            printf("White Queen Side castle\n");

        if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
            printf("Black King Side castle\n");

        if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
            printf("Black Queen Side castle\n");
    }


    CUDA_CALLABLE_MEMBER static void displayBoard(HexaBitBoardPosition *pos)
    {
        BoardPosition p;
        boardHexBBTo088(&p, pos);
        dispBoard(&p);
    }

    CUDA_CALLABLE_MEMBER static void displayCompactMove(CMove move)
    {
        Move move2;
        move2.capturedPiece = (move.getFlags() & CM_FLAG_CAPTURE);
        move2.src = move.getFrom();
        move2.dst = move.getTo();
        displayMoveBB(move2);
    }

    CUDA_CALLABLE_MEMBER static void displayMoveBB(Move move) 
    {
        uint8 r1, c1, r2, c2;
        r1 = (move.src >> 3)+1;
        c1 = (move.src) & 0x7;

        r2 = (move.dst >> 3)+1;
	    c2 = (move.dst) & 0x7;

	    char sep = move.capturedPiece ? '*' : '-';

	    printf("%c%d%c%c%d ", 
                c1+'a', 
                r1, 
			    sep,
                c2+'a', 
                r2);

    }


    static void getCompactMoveString(CMove move, char *str)
    {
        Move move2;
        move2.capturedPiece = (move.getFlags() & CM_FLAG_CAPTURE);
        move2.src = move.getFrom();
        move2.dst = move.getTo();
        getMoveBBString(move2, str);
    }

    static void getMoveBBString(Move move, char *str) 
    {
        uint8 r1, c1, r2, c2;
        r1 = (move.src >> 3)+1;
        c1 = (move.src) & 0x7;

        r2 = (move.dst >> 3)+1;
	    c2 = (move.dst) & 0x7;

	    char sep = move.capturedPiece ? '*' : '-';

	    sprintf(str, "%c%d%c%c%d ", 
                c1+'a', 
                r1, 
			    sep,
                c2+'a', 
                r2);

    }

    //static void board088ToChar(char board[8][8], BoardPosition *pos);
    static void boardCharTo088(BoardPosition *pos, char board[8][8]);

    static void board088ToHexBB(HexaBitBoardPosition *posBB, BoardPosition *pos088);
    //static void boardHexBBTo088(BoardPosition *pos088, HexaBitBoardPosition *posBB);

	// reads a FEN string and sets board and other Game Data accorodingly
	static void readFENString(char fen[], BoardPosition *pos);

	// clears the board (i.e, makes all squares blank)
	static void clearBoard(BoardPosition *pos);
};

#endif