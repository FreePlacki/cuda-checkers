#ifndef MCTS_H
#define MCTS_H

#include "board.cuh"
#include "gamestate.cuh"
#include "helpers.h"
#include "move.cuh"
#include "movegen.cuh"
#include "tree.cuh"
#include <ctime>

// plays the game till the end and reports the result
GameResult playout(GameState gs) {
    MoveList mlist;

    for (;;) {
        GameResult res = game_result(&gs);
        if (res != PENDING)
            return res;

        int is_white = gs.current_player == WHITE;
        generate_moves(&gs.board, is_white, &mlist);
        if (mlist.count == 0)
            return is_white ? BLACK_WON : WHITE_WON;
        Move m = mlist.moves[rand() % mlist.count];
        apply_move(&gs.board, m, is_white, 1);
        next_turn(&gs, is_capture(m));
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
    const int total_playouts = 300'000;
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
            switch (playout(st)) {
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

Move choose_move_cpu(GameState gs, const MoveList *l, double timeout) {
    double t0 = clock();

    Node *root = node_init(NULL, gs);

    node_expand(root, l);

    const int max_iter = 1'000'000;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        if (iter % 100 && (clock() - t0) / CLOCKS_PER_SEC >= timeout * 0.99)
            break;
        // 1. selection
        Node *leaf = node_select_leaf(root);

        if (!node_is_terminal(leaf)) {
            MoveList ml;
            generate_moves(&leaf->gs.board, leaf->gs.current_player == WHITE,
                           &ml);
            // 2. expansion
            node_expand(leaf, &ml);
        }

        // 3. playout
        GameResult res = playout(leaf->gs);

        // 4. backpropagation
        node_backprop(leaf, res);
    }

    // choose the move with most playouts
    int most_pl = 0;
    int best_move_idx = 0;
    Node *best_child;
    for (int i = 0; i < root->children.count; ++i) {
        Node *child = root->children.nodes[i];
        if (child->games_played > most_pl) {
            most_pl = child->games_played;
            best_move_idx = i;
            best_child = child;
        }
    }

    double valuation =
        2.0 * (double)best_child->games_won / (double)best_child->games_played -
        1.0;
    printf("valuation: %lf\nplayouts: %d\ntook: %lf s\n", valuation, iter,
           (double)(clock() - t0) / CLOCKS_PER_SEC);

    node_free(root);
    return l->moves[best_move_idx];
}

#endif /* MCTS_H */
