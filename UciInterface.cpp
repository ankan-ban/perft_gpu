// UCI Interfacing routines
#include "chess.h"



// old code for copying pasting
#if 0
// for CreateThread method.
#include <windows.h>


// handles of the threads
HANDLE hSearchThread;
HANDLE hTimerThread;
move_list_item bestMove;
long maxFixedDepth=30;

// Added for getting the time: Start
unsigned long long t1, t2;
// Added for getting the time: End


long WINAPI searchThread(long param) {
	int depth;
	for(depth=5;depth<maxFixedDepth;depth++) {

		//util::dispBoard(board);

		Search::alphaBeta(depth, -INF, INF, 0);
		bestMove=current_move_line[depth];

		// display the move found at this depth
		//util::dispBoard(board);
		//printfs("\n%d %d %d %d\n", chance, castleflagB, castleflagW, enPassentFlag);
		//printf("info depth %d pv ", depth );
		//util::displayMove(bestMove);

		fflush(stdout);
	}

	// Added for getting time: Start
	struct _timeb timebuffer;
	_ftime( &timebuffer );
	t2 = timebuffer.time*1000 + timebuffer.millitm;

	printf("\nTime Taken: %llu ms\n\n", t2-t1);
	//printf("\n\n\nNPS: %lf\n", totalNodes/((t2-t1)*1000.0f));
	fflush(stdout);
	// Added for getting time: End

	// never normally returns
	return 0;
}

void dispBestMoveFound() {
	printf("bestmove ");
	util::displayMove(bestMove);
	fflush(stdout);
}

long WINAPI timerThread(long searchTime)
{ 
	// create the searchThread and wait for searchTime;
	unsigned long iID;
	hSearchThread = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)searchThread,NULL,0,&iID);

	WaitForSingleObject(hSearchThread, searchTime);
	TerminateThread(hSearchThread, 0);
	
	// display the best move found
	dispBestMoveFound();

	return 0;
}


void search_go(char *params) {
	long search_time=0;
	long x=0;
	long movestogo=0;
	char * pos;
	long maxDepth=0;
	long inc=0;

	pos = strstr(params, "movestogo");
	if(pos) {
		pos+=10;
		sscanf(pos, "%d", &movestogo);
	}

	pos = strstr(params, "wtime");
	if(pos&&chance==WHITE) {
		pos+=6;
		sscanf(pos, "%d", &x);
	}

	pos = strstr(params, "btime");
	if(pos&&chance==BLACK) {
		pos+=6;
		sscanf(pos, "%d", &x);
	}

	pos = strstr(params, "depth");
	if(pos) {
		pos+=6;
		sscanf(pos, "%d", &maxDepth);
		if(maxDepth>2)
			maxFixedDepth=maxDepth;
	}	

	pos = strstr(params, "movetime");
	if(pos) {
		pos+=9;
		sscanf(pos, "%d", &search_time);
	}

	pos = strstr(params, "infinite");
	if(pos) {
		search_time = INF*INF;
	}

	pos = strstr(params, "winc");
	if(pos&&chance==WHITE) {
		pos+=5;
		sscanf(pos, "%d", &inc);
	}
	pos = strstr(params, "binc");
	if(pos&&chance==BLACK) {
		pos+=5;
		sscanf(pos, "%d", &inc);
	}

	if(!search_time) {
			if(movestogo==0) movestogo = 60;
			search_time = x/(movestogo) + inc;
	}

	//printf("searchtime: %d\n", search_time);


	// create the timer thread and return
	// the timer thread will control the searchThread and return after 


	// Added for getting the time: Start
	struct _timeb timebuffer;
	_ftime( &timebuffer );
	t1 = timebuffer.time*1000 + timebuffer.millitm;
	// Added for getting the time: End


	unsigned long iID;
	hTimerThread = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)timerThread,(void *) search_time,0,&iID);

}



// reads a move in algebric notation into a move_list_item
int readMove(const char *move, move_list_item *mi) {
	char r1, c1, r2, c2, q;

	// ignore blank spaces
	while(*move==' ') move++;

	r1 = 7-(move[1]-'1');
	c1 = move[0]-'a';
	r2 = 7-(move[3]-'1');
	c2 = move[2]-'a';
	q = move[4];


	mi->piece = board[r1][c1];
	mi->oldpiece = board[r2][c2];

	// special moves

	// 1. En-Passent
	if(mi->piece%10 == PAWN && mi->oldpiece==0 && c1!=c2) {
		mi->r1=EN_PASSENT;
		mi->r2=chance;
		mi->c1=c1;
		mi->c2=c2;
		return 4;
	}

	// 2. Promotion
	if(mi->piece==WHITE_PAWN && r2==0 || mi->piece==BLACK_PAWN && r2==7) {
		mi->r1 = PROMOTION;
		mi->r2 = util::getPieceCode(q)%10 + (chance+1)*10;
		mi->c1 = c1;
		mi->c2 = c2;
		return 5;
	}

	// 3. Castling
	if(mi->piece%10==KING && abs(c2-c1) > 1) {
		mi->r1=CASTLING;
		if(c2>c1) {		// king side
			if(r1==0)	// black
				mi->c1 = 11;
			else
				mi->c1 = 21;
		}
		else {
			if(r1==0)
				mi->c1 = 12;
			else 
				mi->c1 = 22;
		}
		return 4 ;
	}
	mi->r1=r1;
	mi->c1=c1;
	mi->r2=r2;
	mi->c2=c2;
	return 4;
}

void UciInterface::processCommand() {

	char buffer[8192];
	char *input = buffer;


	// the main game loop
	while(1) {
		input = buffer;
		gets(input);
		if(strstr(input, "ucinewgame")) {
			// new game
			init();
		}
		else if(strstr(input, "uci")) {
			// first command to indicate uci mode

			// send back the IDs
			printf("id name DeltaChess 0.00\n");
			printf("id author Ankan Banerjee\n");

			// send the "uciok" command
			printf("uciok\n");
		}
		else if(strstr(input, "isready")) {
			// send back readyok command
			printf("readyok\n");
		}
		else if(strstr(input, "position")) {
			if(strstr(input, "startpos")) {
				util::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
			}
			else {
				// read the fen string
				input = input + 9;	// skip the "position "
				util::readFENString(input);
			}
			// read the moves
			input = strstr(input, "moves");
			if(input==0) continue;
			input += 6;	// skip the "moves "
			while((*input) && (*input)!='\n') {
				move_list_item move;
				int mlen = readMove(input, &move);
				MoveMaker::makeMove(&move);
				input+=mlen;
				// ignore blank spaces
				while(*input==' ') input++;
			}
				
		}
		else if(strstr(input, "quit")) {
			exit(0);
		}
		else if(strstr(input, "dispboard")) {
			util::dispBoard(board);
		}
		else if(strstr(input, "go")) {
			// ready for some action
			input+=3;
			search_go(input);
		}
		else if(strstr(input, "bench")) {
			util::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
			sscanf(input+6, "%d", &maxFixedDepth);
			printf("Running Bench mark at depth %d. Please wait or type 'quit' to abort...\n", maxFixedDepth);

			// Added for getting the time: Start
			struct _timeb timebuffer;
			_ftime( &timebuffer );
			t1 = timebuffer.time*1000 + timebuffer.millitm;
			// Added for getting the time: End

			unsigned long iID;
			hTimerThread = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)timerThread,(void *) 1000000000,0,&iID);

		}
		else if(strstr(input, "stop")) {
			// stop the current line of search, and display the best move found
			TerminateThread(hTimerThread, 0);
			TerminateThread(hSearchThread, 0);
			// display the best move found
			dispBestMoveFound();

		}
		fflush(stdout);

	}




	/*
	hThread = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)thread2,NULL,0,&iID);

	for(int i=0;i<1000000000;i++)
		if(!(i%1000000))
			printf("In orig thread: %d\n", i/1000000);

	WaitForSingleObject(hThread, 50);
	TerminateThread(hThread, 0);
*/


	exit(0);
}
#endif