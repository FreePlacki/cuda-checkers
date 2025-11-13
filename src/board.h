#ifndef BOARD_H
#define BOARD_H

#include "move.h"

typedef struct MoveList MoveList;

typedef struct {
    // (numbering from left to right, top to bottom, playable squares only)
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
Board flip_perspective(const Board *b);

// compute captured squares mask for a move given the current board.
// returns 1 on success, 0 on error (illegal segment, no captured piece found,
// etc.)
int move_compute_captures(const Board *b, const Move *m,
                          uint32_t *captures_out);

#endif /* BOARD_H */
