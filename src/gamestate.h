
#ifndef GAMESTATE_H
#define GAMESTATE_H

#include "board.h"
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

void init_game(GameState *state);

int seed_game(GameState *gs, FILE *f, FILE *logfile);

void next_turn(GameState *gs, int capture_occured);

GameResult game_result(const GameState *gs);

#endif /* GAMESTATE_H */
