#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"

typedef struct MoveList {
    Move moves[128];
    int count;
} MoveList;

// Generate all legal moves for the player
void generate_moves(const Board *b, int is_white, MoveList *out);

int is_valid_move(const Move *m, const MoveList *l);

void print_movelist(const MoveList *l);

#endif
