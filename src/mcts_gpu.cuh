#ifndef MCTS_GPU_H
#define MCTS_GPU_H

#include "assert.h"
#include "board.cuh"
#include "gamestate.cuh"
#include "move.cuh"
#include "movegen.cuh"
#include "time.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

// https://en.wikipedia.org/wiki/Xorshift
__device__ inline u32 xorshift32(u32 *s) {
    u32 x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

__device__ GameResult playout_device(GameState gs, u32 rand_seed) {
    MoveList mlist;

    for (;;) {
        GameResult res = game_result(&gs);
        if (res != PENDING)
            return res;

        int is_white = gs.current_player == WHITE;
        generate_moves(&gs.board, is_white, &mlist);
        if (mlist.count == 0)
            return is_white ? BLACK_WON : WHITE_WON;

        u32 r = xorshift32(&rand_seed);
        Move m = mlist.moves[r % mlist.count];
        apply_move(&gs.board, m, is_white, 1);
        next_turn(&gs, is_capture(m));
    }
}

__global__ void gpu_playout_kernel(GameState *states, GameResult *results,
                                   int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    GameState s = states[id];
    // floor(2^32 / phi)
    u32 seed = 0x9E3779B9u ^ id;

    results[id] = playout_device(s, seed);
}

int playout_gpu(const GameState *gs, int playouts) {
    GameState *d_states = NULL;
    GameResult *d_results = NULL;

    GameState *h_states = NULL;
    GameResult *h_results = NULL;

    int score = 0;
    cudaEvent_t start_evt, stop_evt;
    float ms = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    // host allocations
    h_states = (GameState *)malloc(playouts * sizeof(GameState));
    h_results = (GameResult *)malloc(playouts * sizeof(GameResult));
    if (!h_states || !h_results) {
        fprintf(stderr, "Host allocation failure.\n");
        goto cleanup;
    }

    // fill host states
    for (int i = 0; i < playouts; i++)
        h_states[i] = *gs;

    // device allocations
    CUDA_CHECK(cudaMalloc((void **)&d_states, playouts * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc((void **)&d_results, playouts * sizeof(GameResult)));

    // copy states to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states, playouts * sizeof(GameState),
                          cudaMemcpyHostToDevice));

    // kernel launch
    {
        int block = 32;
        int grid = (playouts + block - 1) / block;

        CUDA_CHECK(cudaEventRecord(start_evt, 0));
        gpu_playout_kernel<<<grid, block>>>(d_states, d_results, playouts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop_evt, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
        // printf("Kernel time: %.3f ms\n", ms);
    }

    // read results back
    CUDA_CHECK(cudaMemcpy(h_results, d_results, playouts * sizeof(GameResult),
                          cudaMemcpyDeviceToHost));

    // results aggregation
    {
        int is_white = (gs->current_player != WHITE);
        for (int i = 0; i < playouts; i++) {
            GameResult r = h_results[i];
            if (r == WHITE_WON)
                score += is_white ? +1 : -1;
            else if (r == BLACK_WON)
                score += is_white ? -1 : +1;
        }
    }

cleanup:
    if (d_states)
        cudaFree(d_states);
    if (d_results)
        cudaFree(d_results);
    if (h_states)
        free(h_states);
    if (h_results)
        free(h_results);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    return score;
}

Move choose_move_flat_gpu(const GameState *gs, const MoveList *root_moves) {
    assert(root_moves->count > 0);
    if (root_moves->count == 1)
        return root_moves->moves[0];

    const int total_playouts = 30'000'000;
    const int playouts = total_playouts / root_moves->count;

    int best_score = -playouts;
    int best_idx = 0;

    double t0 = clock();
    for (int i = 0; i < root_moves->count; ++i) {

        GameState next = *gs;
        apply_move(&next.board, root_moves->moves[i],
                   gs->current_player == WHITE, 1);
        next_turn(&next, is_capture(root_moves->moves[i]));

        int score = playout_gpu(&next, playouts);

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    printf("valuation: %.4f\nplayouts: %d\ntook: %lf s\n",
           (double)best_score / playouts, total_playouts,
           (double)(clock() - t0) / CLOCKS_PER_SEC);

    return root_moves->moves[best_idx];
}

#define BATCH 10'000 // max number of leaf simulations per GPU round
Move choose_move_gpu(GameState gs, const MoveList *root_moves, double timeout) {
    assert(root_moves->count > 0);
    if (root_moves->count == 1)
        return root_moves->moves[0];

    double t0 = clock();

    Node *root = node_init(NULL, gs);
    node_expand(root, root_moves);

    GameState *d_states, *h_states;
    GameResult *d_results, *h_results;

    Node *best = NULL;
    Move best_move;
    int best_idx = 0;
    int total_playouts = 0;
    double valuation;

    CUDA_CHECK(cudaMalloc((void **)&d_states, BATCH * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc((void **)&d_results, BATCH * sizeof(GameResult)));

    h_states = (GameState *)malloc(BATCH * sizeof(GameState));
    h_results = (GameResult *)malloc(BATCH * sizeof(GameResult));
    if (!h_states || !h_results)
        exit(1);

    for (int i = 0; i < 1'000; ++i) {
        Node *leaf = node_select_leaf(root);

        if (!node_is_terminal(leaf)) {
            MoveList ml;
            generate_moves(&leaf->gs.board, leaf->gs.current_player == WHITE,
                           &ml);
            node_expand(leaf, &ml);
            GameResult r = playout(leaf->gs);
            node_backprop(leaf, r);
        }
    }

    for (int iter = 0;; ++iter) {
        if ((clock() - t0) / CLOCKS_PER_SEC >= 0.98 * timeout)
            break;
        Node *leaf_batch[BATCH];

        int count = 0;
        for (count = 0; count < BATCH; ++count) {
            Node *leaf = node_select_leaf(root);

            if (!node_is_terminal(leaf)) {
                MoveList ml;
                generate_moves(&leaf->gs.board,
                               leaf->gs.current_player == WHITE, &ml);

                node_expand(leaf, &ml);
            }

            leaf_batch[count] = leaf;
            h_states[count] = leaf->gs;
        }

        CUDA_CHECK(cudaMemcpy(d_states, h_states, count * sizeof(GameState),
                              cudaMemcpyHostToDevice));

        int block = 32;
        int grid = (count + block - 1) / block;
        gpu_playout_kernel<<<grid, block>>>(d_states, d_results, count);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_results, d_results, count * sizeof(GameResult),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < count; i++)
            node_backprop(leaf_batch[i], h_results[i]);

        total_playouts += count;
    }

    // choose child with most visits
    for (int i = 0; i < root->children.count; i++) {
        Node *ch = root->children.nodes[i];
        if (!best || ch->games_played > best->games_played) {
            best = ch;
            best_idx = i;
        }
    }

    valuation =
        2.0 * (double)best->games_won / (double)best->games_played - 1.0;
    printf("valuation: %lf\nplayouts: %d\ntook: %lf s\n", valuation,
           total_playouts, (double)(clock() - t0) / CLOCKS_PER_SEC);

    best_move = root_moves->moves[best_idx];
    node_free(root);

cleanup:
    if (d_states)
        cudaFree(d_states);
    if (d_results)
        cudaFree(d_results);
    if (h_states)
        free(h_states);
    if (h_results)
        free(h_results);
    return best_move;
}

#endif /* MCTS_GPU_H */
