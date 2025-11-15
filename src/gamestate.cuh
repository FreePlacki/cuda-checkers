#ifndef GAMESTATE_H
#define GAMESTATE_H

#include "board.cuh"
#include "move.cuh"
#include <stdio.h>

typedef enum {
    BLACK,
    WHITE,
} Player;

typedef enum {
    BLACK_WON,
    WHITE_WON,
    DRAW,
    PENDING,
} GameResult;

typedef struct {
    Board board;
    u8 current_player;
    u8 no_capture; // number of conescutive non-capture moves
} GameState;

void init_game(GameState *state) {
    state->current_player = BLACK;
    state->no_capture = 0;
    init_board(&state->board);
}

__host__ __device__
void next_turn(GameState *gs, int capture_occured) {
    gs->current_player = !gs->current_player;
    if (capture_occured)
        gs->no_capture = 0;
    else
        gs->no_capture++;
}


int seed_game(GameState *gs, FILE *f, FILE *logfile) {
    char buf[MOVE_STR_MAX];

    Move m;
    while (fgets(buf, MOVE_STR_MAX, f)) {
        if (!parse_move(buf, &m))
            return 0;
        apply_move(&gs->board, &m, 1);
        next_turn(gs, is_capture(&m));

        move_to_str(&m, buf);
        if (logfile)
            fprintf(logfile, "%s\n", buf);
    }
    return 1;
}

#define MAX_NOCAPTURE 50
__host__ __device__
GameResult game_result(const GameState *gs) {
    if (gs->no_capture >= MAX_NOCAPTURE)
        return DRAW;
    if (gs->board.white == 0u)
        return BLACK_WON;
    if (gs->board.black == 0u)
        return WHITE_WON;
    return PENDING;
}

#endif /* GAMESTATE_H */
