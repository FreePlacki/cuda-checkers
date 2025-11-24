#ifndef MCTS_H
#define MCTS_H

#include "board.cuh"
#include "gamestate.cuh"
#include "movegen.cuh"

// plays the game till the end and reports the result
GameResult playout(const GameState *gs) {
    MoveList mlist;
    GameState state;

    memcpy(&state, gs, sizeof(GameState));

    for (;;) {
        int is_white = state.current_player == WHITE;
        generate_moves(&state.board, is_white, &mlist);
        if (mlist.count == 0)
            return is_white ? BLACK_WON : WHITE_WON;
        Move m = mlist.moves[rand() % mlist.count];
        apply_move(&state.board, m, is_white, 1);
        next_turn(&state, is_capture(m));
        GameResult res = game_result(&state);
        if (res != PENDING)
            return res;
    }
}

Move choose_move_rand(const GameState *gs, const MoveList *l) {
    int r = rand() % l->count;
    return l->moves[r];
}

Move choose_move_flat_cpu(const GameState *gs, const MoveList *l) {
    if (l->count == 1)
        return l->moves[0];

    int is_white = gs->current_player == WHITE;
    const int total_playouts = 1'000'000;
    const int playouts = total_playouts / l->count;
    int best_res = -playouts;
    int best_idx = 0;

    double t0 = clock();
    for (int i = 0; i < l->count; ++i) {
        int score = 0;
        GameState st = *gs;
        apply_move(&st.board, l->moves[i], st.current_player == WHITE, 1);
        next_turn(&st, is_capture(l->moves[i]));

        for (int j = 0; j < playouts; ++j) {
            switch (playout(&st)) {
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
           (double)best_res / playouts, total_playouts,
           (double)(clock() - t0) / CLOCKS_PER_SEC);

    return l->moves[best_idx];
}

#endif /* MCTS_H */
