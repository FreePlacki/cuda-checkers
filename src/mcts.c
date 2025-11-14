#include "mcts.h"
#include "board.h"
#include "gamestate.h"
#include "helpers.h"
#include "movegen.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

GameResult playout(const GameState *gs) {
    MoveList mlist;
    GameState state;

    memcpy(&state, gs, sizeof(GameState));

    for (;;) {
        generate_moves(&state.board, state.current_player == WHITE, &mlist);
        if (mlist.count == 0)
            return state.current_player == WHITE ? BLACK_WON : WHITE_WON;
        Move m = mlist.moves[rand() % mlist.count];
        apply_move(&state.board, &m, 1);
        next_turn(&state, is_capture(&m));
        GameResult res = game_result(&state);
        if (res != PENDING)
            return res;
    }
}

Move choose_move_rand(const GameState *gs, const MoveList *l) {
    int r = rand() % l->count;
    return l->moves[r];
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
Move choose_move(const GameState *gs, const MoveList *l) {
    if (l->count == 1)
        return l->moves[0];

    int is_white = gs->current_player == WHITE;
    const int playouts = 10000;
    int best_res = -playouts;
    int best_idx = 0;

    double t0 = now();
    for (int i = 0; i < l->count; ++i) {
        int score = 0;
        for (int j = 0; j < playouts; ++j) {
            switch (playout(gs)) {
            case BLACK_WON:
                score += is_white ? -1 : 1;
                break;
            case WHITE_WON:
                score += is_white ? 1 : -1;
                break;
            case DRAW:
                break;
            case PENDING:
                assert(0 && "unreachable");
            }
        }
        if (score > best_res) {
            best_res = score;
            best_idx = i;
        }
    }
    printf("valuation: %lf\nplayouts: %d\ntook: %lf s\n",
           (double)best_res / playouts, playouts, now() - t0);

    return l->moves[best_idx];
}
