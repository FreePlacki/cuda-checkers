#ifndef BOARD_H
#define BOARD_H

#include "move.h"

typedef struct MoveList MoveList;

/*
Board representation:

  11  05  31  25 
10  04  30  24 
  03  29  23  17 
02  28  22  16 
  27  21  15  09 
26  20  14  08 
  19  13  07  01 
18  12  06  00

Notice that every piece (except ones on edges) can move +1 or +7 up
and -1 or -7 down (mod 32)
Source: https://3dkingdoms.com/checkers/bitboards.htm#A1
*/

extern const int idx_to_board[32];

typedef struct {
    // positions of white pieces
    u32 white;
    // positions of black pieces
    u32 black;
    // positions of kings, either white or black
    u32 kings;
} Board;

void init_board(Board *b);
void print_board(const Board *state, const MoveList *mlist,
                 const Move *last_move);
void apply_move(Board *b, const Move *m);

// compute captured squares mask for a move given the current board.
// returns 1 on success, 0 on error (illegal segment, no captured piece found,
// etc.)
int move_compute_captures(const Board *b, const Move *m,
                          uint32_t *captures_out);

#endif /* BOARD_H */
