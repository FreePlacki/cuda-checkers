#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"

typedef struct {
    Move moves[128];
    int count;
} MoveList;

// IMPORTANT: for the functions to work, the Board has to be normalized to 
// have the player on move be on top (use flip if necessary)

// Generate all non-capture moves or single-capture moves
void generate_single(const Board *b, int is_white, MoveList *out, u32 mask);

// Generate all legal moves for the player
void generate_moves(const Board *b, int is_white, MoveList *out);

int is_valid_move(const Move *m, const MoveList *l);

void flip_moves(MoveList *l);

void print_movelist(const MoveList *l);

#endif
