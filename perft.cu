// a simple 0x88 move generater / perft tool

//#include "chess.h"
#include "MoveGenerator088.h"

// max no of moves possible for a given board position (this can be as large as 218 ?)
// e.g, test this FEN string "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"
#define MAX_MOVES 256

// routines to make a move on the board and to undo it

__host__ __device__ void makeMove(BoardPosition *pos, Move move)
{
    uint8 piece = PIECE(pos->board[move.src]);
    uint32 chance = pos->chance;

    pos->board[move.dst] = pos->board[move.src];
    pos->board[move.src] = EMPTY_SQUARE;

    if (move.flags)
    {
        // special  moves

        // 1. Castling: update the rook position too
        if(move.flags == CASTLE_KING_SIDE)
        {
            if (chance == BLACK)
            {
                pos->board[0x77] = EMPTY_SQUARE; 
                pos->board[0x75] = COLOR_PIECE(BLACK, ROOK);
            }
            else
            {
                pos->board[0x07] = EMPTY_SQUARE; 
                pos->board[0x05] = COLOR_PIECE(WHITE, ROOK);
            }
                        
        }
        else if (move.flags == CASTLE_QUEEN_SIDE)
        {
            if (chance == BLACK)
            {
                pos->board[0x70] = EMPTY_SQUARE; 
                pos->board[0x73] = COLOR_PIECE(BLACK, ROOK);
            }
            else
            {
                pos->board[0x00] = EMPTY_SQUARE; 
                pos->board[0x03] = COLOR_PIECE(WHITE, ROOK);
            }
        }

        // 2. en-passent: clear the captured piece
        else if (move.flags == EN_PASSENT)
        {
            pos->board[INDEX088(RANK(move.src), pos->enPassent - 1)] = EMPTY_SQUARE;
        }

        // 3. promotion: update the pawn to promoted piece
        else if (move.flags == PROMOTION_QUEEN)
        {
            pos->board[move.dst] = COLOR_PIECE(chance, QUEEN);
        }
        else if (move.flags == PROMOTION_ROOK)
        {
            pos->board[move.dst] = COLOR_PIECE(chance, ROOK);
        }
        else if (move.flags == PROMOTION_KNIGHT)
        {
            pos->board[move.dst] = COLOR_PIECE(chance, KNIGHT);
        }
        else if (move.flags == PROMOTION_BISHOP)
        {
            pos->board[move.dst] = COLOR_PIECE(chance, BISHOP);
        }
    }

    // update game state variables
    pos->enPassent = 0;

    if (piece == KING)
    {
        if (chance == BLACK)
        {
            pos->blackCastle = 0;
        }
        else
        {
            pos->whiteCastle = 0;
        }
    }
    else if (piece == ROOK)
    {
        if (chance == BLACK)
        {
            if (move.src == 0x77)
                pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (move.src == 0x70)
                pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
        else
        {
            if (move.src == 0x7)
                pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (move.src == 0x0)
                pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
    }
    else if ((piece == PAWN) && (abs(RANK(move.dst) - RANK(move.src)) == 2))
    {
        pos->enPassent = FILE(move.src) + 1;
    }

    // clear the appriopiate castle flag if a rook is captured
    if (PIECE(move.capturedPiece) == ROOK)
    {
        if (chance == BLACK)
        {
            if (move.dst == 0x7)
                pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (move.dst == 0x0)
                pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
        else
        {
            if (move.dst == 0x77)
                pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
            else if (move.dst == 0x70)
                pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
        }
    }

    // flip the chance
    pos->chance = !chance;
}

__host__ __device__ void undoMove(BoardPosition *pos, Move move, uint8 bc, uint8 wc, uint8 enPassent)
{
    pos->board[move.src] = pos->board[move.dst];    
    pos->board[move.dst] = move.capturedPiece;

    pos->blackCastle = bc;
    pos->whiteCastle = wc;
    pos->enPassent = enPassent;
    pos->chance = !pos->chance;
}


// recursive perft search
__device__ __host__ uint64 perft(BoardPosition *pos, int depth)
{
    Move moves[MAX_MOVES];
    uint64 childPerft = 0;

    uint32 nMoves = MoveGenerator::generateMoves(pos, moves);
    if (depth == 1)
    {
        return nMoves;
    }

    for (uint32 i = 0; i < nMoves; i++)
    {
        BoardPosition temp = *pos;
        makeMove(&temp, moves[i]);
        childPerft += perft(&temp, depth - 1);
    }
    return childPerft;
}




///////////////////------------------------------ GPU implementation ---------------------------------//

// perft search
__global__ void perft_gpu(BoardPosition *position, uint64 *generatedMoves, int depth, uint32 nodeEstimate)
{
    // exctact one element of work
    BoardPosition *pos = &(position[threadIdx.x]);
    uint64 *moveCounter = &(generatedMoves[threadIdx.x]);
    

    // TODO: check if keeping this local variable is ok
    Move moves[MAX_MOVES];  // huge structure in thread local memory
    uint64 childPerft = 0;

    uint32 nMoves = MoveGenerator::generateMoves(pos, moves);

    if (depth == 1 || nMoves == 0)
    {
        *moveCounter = nMoves;
        return;
    }


    if (nodeEstimate < 1000000)
    {
        cudaStream_t childStream;
        cudaStreamCreateWithFlags(&childStream, cudaStreamNonBlocking);

        BoardPosition *childBoards;
        uint64 *child_perfts;
        int hr;
        hr = cudaMalloc(&childBoards, sizeof(BoardPosition) * nMoves);
        
        //if (hr != 0)
        //    printf("error in malloc at depth %d\n", depth);
        hr = cudaMalloc(&child_perfts, sizeof(uint64) * nMoves);
        //if (hr != 0)
        //    printf("error in sedond malloc at depth %d\n", depth);
         

        for (uint32 i = 0; i < nMoves; i++)
        {
            childBoards[i] = *pos;
            makeMove(&childBoards[i], moves[i]);
            child_perfts[i] = 0;
        }

        nodeEstimate *= nMoves;
        perft_gpu<<<1, nMoves, 0, childStream>>> (childBoards, child_perfts, depth-1, nodeEstimate);
        cudaDeviceSynchronize();

        for (uint32 i = 0; i < nMoves; i++)
        {
            childPerft += child_perfts[i];
        }

        cudaFree(childBoards);
        cudaFree(child_perfts);
        cudaStreamDestroy(childStream);
    }
    else
    {
        // call recursively in same thread
        for (uint32 i = 0; i < nMoves; i++)
        {
            BoardPosition temp = *pos;
            makeMove(&temp, moves[i]);
            childPerft += perft(&temp, depth - 1);
        }
    }
    
    
    *moveCounter = childPerft;
}






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
  void start(cudaStream_t s = 0) { cudaEventRecord(mStart, s); 
                                   mStarted = true; mStopped = false; }
  void stop(cudaStream_t s = 0)  { assert(mStarted);
                                   cudaEventRecord(mStop, s); 
                                   mStarted = false; mStopped = true; }
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
double gTime;
#define START_TIMER { \
    LARGE_INTEGER count1, count2, freq; \
    QueryPerformanceFrequency (&freq);  \
    QueryPerformanceCounter(&count1);

#define STOP_TIMER \
    QueryPerformanceCounter(&count2); \
    gTime = ((double)(count2.QuadPart-count1.QuadPart)*1000.0)/freq.QuadPart; \
    }
// for timing CPU code : end


int main()
{
    BoardPosition testBoard;

    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results

    Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions

    //Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)
    //Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!

    

    //Move moves[MAX_MOVES];
    //uint32 nMoves = MoveGenerator::generateMoves(&testBoard, moves);
    //printf("\nMoves generated: %d\n", nMoves);


    printf("\nEnter FEN String: \n");
    char fen[1024];
    gets(fen);
    Utils::readFENString(fen, &testBoard); // start.. 20 positions
    Utils::dispBoard(&testBoard);

    int depth;
    printf("\nEnter depth: ");
    scanf("%d", &depth);

    //for (int depth=1;depth<7;depth++)
    {
        
        uint64 leafNodes;
        
        START_TIMER
        leafNodes = perft(&testBoard, depth);
        STOP_TIMER
        printf("\nPerft %d: %llu,   ", depth, leafNodes);
        printf("Time taken: %g seconds, nps: %llu\n", gTime/1000.0, (uint64) ((leafNodes/gTime)*1000.0));
        


        // try the same thing on GPU
        int hr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, depth);
        printf("cudaDeviceSetLimit returned %d\n", hr);

        hr = cudaDeviceSetLimit(cudaLimitStackSize, 4*1024);
        printf("cudaDeviceSetLimit stack size returned %d\n", hr);

        BoardPosition *gpuBoard;
        uint64 *gpu_perft;
        cudaMalloc(&gpuBoard, sizeof(BoardPosition));
        cudaMalloc(&gpu_perft, sizeof(uint64));
        hr = cudaMemcpy(gpuBoard, &testBoard, sizeof(BoardPosition), cudaMemcpyHostToDevice);
        printf("cudaMemcpyHostToDevice returned %d\n", hr);
        EventTimer gputime;

        gputime.start();
        perft_gpu <<<1, 1>>> (gpuBoard, gpu_perft, depth, 1);
        gputime.stop();
        printf("host side launch returned: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaDeviceSynchronize();

        uint64 res;
        hr = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
        printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString( (cudaError_t) hr));

        printf("\nGPU Perft %d: %llu,   ", depth, res);
        printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed()/1000.0, (uint64) ((res/gputime.elapsed())*1000.0));

        cudaFree(gpuBoard);
	}

    return 0;
}