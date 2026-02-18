#include "cuda_runtime.h"
#include <chrono>


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


    // convert quad bitboard to 088 board
    CUDA_CALLABLE_MEMBER static void boardQuadBBTo088(BoardPosition *pos088, QuadBitBoard *qbb, GameState *gs)
    {
        memset(pos088, 0, sizeof(BoardPosition));

        for (uint8 i = 0; i < 64; i++)
        {
            uint8 rank = i >> 3;
            uint8 file = i & 7;
            uint8 index088 = INDEX088(rank, file);

            // extract piece type from quad encoding
            uint8 piece = 0;
            if (qbb->bb[1] & BIT(i)) piece |= 1;
            if (qbb->bb[2] & BIT(i)) piece |= 2;
            if (qbb->bb[3] & BIT(i)) piece |= 4;

            if (piece)
            {
                uint8 color = (qbb->bb[0] & BIT(i)) ? BLACK : WHITE;
                pos088->board[index088] = COLOR_PIECE(color, piece);
            }
        }

        pos088->chance = gs->chance;
        pos088->blackCastle = gs->blackCastle;
        pos088->whiteCastle = gs->whiteCastle;
        pos088->enPassent = gs->enPassent;
        pos088->halfMoveCounter = gs->halfMoveCounter;
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


    CUDA_CALLABLE_MEMBER static void displayBoard(QuadBitBoard *pos, GameState *gs)
    {
        BoardPosition p;
        boardQuadBBTo088(&p, pos, gs);
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
        r1 = (move.src >> 3) + 1;
        c1 = (move.src) & 0x7;

        r2 = (move.dst >> 3) + 1;
        c2 = (move.dst) & 0x7;

        char sep = move.capturedPiece ? '*' : '-';

        printf("%c%d%c%c%d ",
            c1 + 'a',
            r1,
            sep,
            c2 + 'a',
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
        r1 = (move.src >> 3) + 1;
        c1 = (move.src) & 0x7;

        r2 = (move.dst >> 3) + 1;
        c2 = (move.dst) & 0x7;

        char sep = move.capturedPiece ? '*' : '-';

        sprintf(str, "%c%d%c%c%d ",
            c1 + 'a',
            r1,
            sep,
            c2 + 'a',
            r2);

    }

    //static void board088ToChar(char board[8][8], BoardPosition *pos);
    static void boardCharTo088(BoardPosition *pos, char board[8][8]);

    static void board088ToQuadBB(QuadBitBoard *qbb, GameState *gs, BoardPosition *pos088);

    // reads a FEN string and sets board and other Game Data accorodingly
    static void readFENString(const char fen[], BoardPosition *pos);

    // clears the board (i.e, makes all squares blank)
    static void clearBoard(BoardPosition *pos);
};


class EventTimer {
public:
    EventTimer() : mStarted(false), mStopped(false) {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
    }
    ~EventTimer() {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mStop);
    }
    void start(cudaStream_t s = 0) {
        cudaEventRecord(mStart, s);
        mStarted = true; mStopped = false;
    }
    void stop(cudaStream_t s = 0)  {
        assert(mStarted);
        cudaEventRecord(mStop, s);
        mStarted = false; mStopped = true;
    }
    float elapsed() {
        assert(mStopped);
        if (!mStopped) return 0;
        cudaEventSynchronize(mStop);
        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, mStart, mStop);
        return elapsed;
    }

private:
    bool mStarted, mStopped;
    cudaEvent_t mStart, mStop;
};

// for timing CPU code : start
static double gTime;
#define START_TIMER { \
    auto t_start = std::chrono::high_resolution_clock::now();

#define STOP_TIMER \
    auto t_end = std::chrono::high_resolution_clock::now();\
    gTime = std::chrono::duration<double>(t_end-t_start).count();}
// for timing CPU code : end


static void hugeMemset(void *data, uint64 size)
{
    uint8 *mem = (uint8*)data;
    const uint64 c4G = 4ull * 1024 * 1024 * 1024;

    while (size > c4G)
    {
        cudaMemset(mem, 0, c4G);

        mem += c4G;
        size -= c4G;
    }

    cudaMemset(mem, 0, size);
}
