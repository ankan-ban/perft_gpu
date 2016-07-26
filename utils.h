#include "cuda_runtime.h"

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

    static void board088ToHexBB(HexaBitBoardPosition *posBB, BoardPosition *pos088);
    //static void boardHexBBTo088(BoardPosition *pos088, HexaBitBoardPosition *posBB);

    // reads a FEN string and sets board and other Game Data accorodingly
    static void readFENString(char fen[], BoardPosition *pos);

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
    clock_t start, end; \
    start = clock();

#define STOP_TIMER \
    end = clock(); \
    gTime = (double)(end - start) / 1000.0; }
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
