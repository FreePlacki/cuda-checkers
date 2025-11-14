#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "gamestate.h"

// plays the game till the end and reports the result
GameResult playout(const GameState *gs);

Move choose_move_rand(const GameState *gs, const MoveList *l);
Move choose_move(const GameState *gs, const MoveList *l);

#endif /* MCTS_H */
