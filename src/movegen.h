#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"

typedef struct {
    Move moves[128];
    int count;
} MoveList;

// Generate all legal moves for the player
int generate_moves(const Board *b, int is_white, MoveList *out);

#endif
