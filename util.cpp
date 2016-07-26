#include "chess.h"


// Utilsity functions for reading FEN String, EPD file, displaying board, etc

// gets the numeric code of the piece represented by a character
uint8 Utils::getPieceCode(char piece) 
{
	switch(piece) {
	case 'p':
		return COLOR_PIECE(BLACK, PAWN);
	case 'n':
		return COLOR_PIECE(BLACK, KNIGHT);
	case 'b':
		return COLOR_PIECE(BLACK, BISHOP);
	case 'r':
		return COLOR_PIECE(BLACK, ROOK);
	case 'q':
		return COLOR_PIECE(BLACK, QUEEN);
	case 'k':
		return COLOR_PIECE(BLACK, KING);
	case 'P':
		return COLOR_PIECE(WHITE, PAWN);
	case 'N':
		return COLOR_PIECE(WHITE, KNIGHT);
	case 'B':
		return COLOR_PIECE(WHITE, BISHOP);
	case 'R':
		return COLOR_PIECE(WHITE, ROOK);
	case 'Q':
		return COLOR_PIECE(WHITE, QUEEN);
	case 'K':
		return COLOR_PIECE(WHITE, KING);
	default:
		return EMPTY_SQUARE;

	}
}

/* 
  Format of board in text file (e.g. Starting Position):

  rnbqkbnr
  pppppppp
  ........
  ........
  ........
  ........
  PPPPPPPP
  RNBQKBNR
*/

// methods to read a board from text file
void Utils::readBoardFromFile(char filename[], char board[8][8]) {
	FILE * fp = fopen(filename, "r");
	char buf[100];
	for(int i=0;i<8;i++) {
		fscanf(fp, "%s", buf);
		for(int j=0;j<8;j++)
			board[i][j] = buf[j];
	}
	fclose(fp);
}

// convert to char board
void Utils::boardCharTo088(BoardPosition *pos, char board[8][8])
{
    int i, j;
    int index088 = 0;

    for (i = 7; i >= 0; i--)
    {
        for (j = 0; j < 8; j++)
        {
            char piece = board[i][j];
            pos->board[index088] = getPieceCode(piece);
            index088++;
        }
        // skip 8 cells of padding
        index088 += 8;
    }
}

void Utils::readBoardFromFile(char filename[], BoardPosition *pos) 
{
    char board[8][8];
    readBoardFromFile(filename, board);
    boardCharTo088(pos, board);
}

// convert 088 board to hex bit board
void Utils::board088ToHexBB(HexaBitBoardPosition *posBB, BoardPosition *pos088)
{
    memset(posBB, 0, sizeof(HexaBitBoardPosition));

    for (uint8 i=0;i<64;i++)
    {
        uint8 rank = i >> 3;
        uint8 file = i & 7;
        uint8 index088 = INDEX088(rank, file);
        uint8 colorpiece = pos088->board[index088];
        if (colorpiece != EMPTY_SQUARE)
        {
            uint8 color = COLOR(colorpiece);
            uint8 piece = PIECE(colorpiece);
            if (color == WHITE)
            {
                posBB->whitePieces |= BIT(i);
            }
            switch (piece)
            {
                case PAWN:
                    posBB->pawns |= BIT(i);
                    break;
                case KNIGHT:
                    posBB->knights |= BIT(i);
                    break;
                case BISHOP:
                    posBB->bishopQueens |= BIT(i);
                    break;
                case ROOK:
                    posBB->rookQueens |= BIT(i);
                    break;
                case QUEEN:
                    posBB->bishopQueens |= BIT(i);
                    posBB->rookQueens |= BIT(i);
                    break;
                case KING:
                    posBB->kings |= BIT(i);
                    break;
            }
        }
    }

    posBB->chance = pos088->chance;
    posBB->blackCastle = pos088->blackCastle;
    posBB->whiteCastle = pos088->whiteCastle;
    posBB->enPassent = pos088->enPassent;
    posBB->halfMoveCounter = pos088->halfMoveCounter;
}


void Utils::clearBoard(BoardPosition *pos)
{
	for(int i=0;i<8;i++)
		for (int j=0;j<8;j++)
			pos->board[INDEX088(i, j)] = EMPTY_SQUARE;
}

// displays a move object
void Utils::displayMove(Move move) 
{
	char dispString[10];

    uint8 r1, c1, r2, c2;
    r1 = RANK(move.src)+1;
    c1 = FILE(move.src);

    r2 = RANK(move.dst)+1;
	c2 = FILE(move.dst);

	char sep = move.capturedPiece ? '*' : '-';

	sprintf(dispString, "%c%d%c%c%d ", 
            c1+'a', 
            r1, 
			sep,
            c2+'a', 
            r2);

    printf(dispString);
}

// reads a FEN string into the given BoardPosition object

/*
Reference: (Wikipedia)

A FEN record contains 6 fields. The separator between fields is a space. The fields are:

   1. Piece placement (from white's perspective). Each rank is described, starting with rank 8 and ending with rank 1; within each rank, the contents of each square are described from file a through file h. White pieces are designated using upper-case letters ("KQRBNP"), Black by lowercase ("kqrbnp"). Blank squares are noted using digits 1 through 8 (the number of blank squares), and "/" separate ranks.
   2. Active color. "w" means white moves next, "b" means black.
   3. Castling availability. If neither side can castle, this is "-". Otherwise, this has one or more letters: "K" (White can castle kingside), "Q" (White can castle queenside), "k" (Black can castle kingside), and/or "q" (Black can castle queenside).
   4. En passant target square in algebraic notation. If there's no en passant target square, this is "-". If a pawn has just made a 2-square move, this is the position "behind" the pawn.
   5. Halfmove clock: This is the number of halfmoves since the last pawn advance or capture. This is used to determine if a draw can be claimed under the fifty move rule.
   6. Fullmove number: The number of the full move. It starts at 1, and is incremented after Black's move.

*/
void Utils::readFENString(char fen[], BoardPosition *pos) 
{
	int i, j;
	char curChar;
	int row = 0, col = 0;

    memset(pos, 0, sizeof(BoardPosition));

	// 1. read the board
	for(i=0;fen[i];i++) 
    {
		curChar = fen[i];


		if(curChar=='/'||curChar=='\\') 
        {
			row++; col=0;
		}
		else if(curChar >= '1' && curChar <= '8') 
        {	// blank squares
			for(j = 0; j < curChar - '0'; j++)
            {
                pos->board[INDEX088(7-row, col)] = getPieceCode(curChar);
				col++;
            }
		}
		else if(curChar=='k'||curChar=='q'||curChar=='r'||curChar=='b'||curChar=='n'||curChar=='p' ||
				curChar=='K'||curChar=='Q'||curChar=='R'||curChar=='B'||curChar=='N'||curChar=='P') 
        {
            pos->board[INDEX088(7 - row, col)] = getPieceCode(curChar);
			col++;
		}

		if(row >= 7 && col == 8) break;		// done with the board, set the flags now
		
	}

	i++;
	
	// 2. read the chance
	while(fen[i]==' ') 
        i++;

	if(fen[i]=='b' || fen[i]=='B') 
    {
        pos->chance = BLACK; 
    }
	else 
    {
        pos->chance = WHITE;
    }

	i++;

	// 3. read the castle flags
    pos->whiteCastle = pos->blackCastle = 0;

	while(fen[i]==' ') 
        i++;

	while(fen[i]!= ' ') {
		switch(fen[i]) {
		case 'k':
            pos->blackCastle |= CASTLE_FLAG_KING_SIDE;
			break;
		case 'q':
            pos->blackCastle |= CASTLE_FLAG_QUEEN_SIDE;
			break;
		case 'K':
            pos->whiteCastle |= CASTLE_FLAG_KING_SIDE;
			break;
		case 'Q':
            pos->whiteCastle |= CASTLE_FLAG_QUEEN_SIDE;
			break;
		}
		i++;
	}

	// 4. read en-passent flag
    pos->enPassent = 0;

	while(fen[i]==' ') 
        i++;

	if(fen[i] >= 'a' && fen[i] <= 'h')
		pos->enPassent = fen[i] - 'a' + 1;
	
	while(fen[i]!=' ' && fen[i]) 
        i++;
	
	//TODO: 5. read the half-move and the full move clocks

}