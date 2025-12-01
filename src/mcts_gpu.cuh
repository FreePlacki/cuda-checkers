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

    h_states = (GameState *)malloc(playouts * sizeof(GameState));
    h_results = (GameResult *)malloc(playouts * sizeof(GameResult));
    if (!h_states || !h_results) {
        fprintf(stderr, "Host allocation failure.\n");
        goto cleanup;
    }

    for (int i = 0; i < playouts; i++)
        h_states[i] = *gs;

    CUDA_CHECK(cudaMalloc((void **)&d_states, playouts * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc((void **)&d_results, playouts * sizeof(GameResult)));

    CUDA_CHECK(cudaMemcpy(d_states, h_states, playouts * sizeof(GameState),
                          cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaMemcpy(h_results, d_results, playouts * sizeof(GameResult),
                          cudaMemcpyDeviceToHost));

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

#define BATCH 1000 // max number of leaf simulations per GPU round
#define PLAYOUTS_PER_NODE 32
Move choose_move_gpu(GameState gs, const MoveList *root_moves, double timeout) {
    assert(root_moves && root_moves->count > 0);
    if (root_moves->count == 1)
        return root_moves->moves[0];

    double t0 = clock();

    const size_t max_total = BATCH * PLAYOUTS_PER_NODE;

    Node *root = node_init(NULL, gs);
    node_expand(root, root_moves);

    // Device buffers (2 buffers for double buffering)
    GameState *d_states[2];
    GameResult *d_results[2];
    // Host pinned buffers
    GameState *h_states[2];
    GameResult *h_results[2];

    // Host-side arrays of leaf pointers per buffer
    Node **leaf_batch_host[2];

    cudaStream_t streams[2];

    int cur = 0;            // index of buffer we are filling now (0 or 1)
    int prev = -1;          // index of buffer currently executing on GPU (or -1 none)
    size_t prev_total = 0;  // number of playouts in prev buffer (for copying back/backprop)

    int total_playouts = 0;
    int iterations = 0;

    // choose best child (most visits)
    Node *best = NULL;
    int best_idx = 0;
    Move best_move;

    double valuation;

    CUDA_CHECK(cudaMalloc((void **)&d_states[0], max_total * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc((void **)&d_states[1], max_total * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc((void **)&d_results[0], max_total * sizeof(GameResult)));
    CUDA_CHECK(cudaMalloc((void **)&d_results[1], max_total * sizeof(GameResult)));

    CUDA_CHECK(cudaHostAlloc((void **)&h_states[0], max_total * sizeof(GameState), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_states[1], max_total * sizeof(GameState), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_results[0], max_total * sizeof(GameResult), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_results[1], max_total * sizeof(GameResult), cudaHostAllocDefault));

    leaf_batch_host[0] = (Node **)malloc(max_total * sizeof(Node *));
    leaf_batch_host[1] = (Node **)malloc(max_total * sizeof(Node *));
    if (!leaf_batch_host[0] || !leaf_batch_host[1]) {
        fprintf(stderr, "Host allocation failure for leaf_batch_host\n");
        goto cleanup;
    }

    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    for (int i = 0; i < 300; ++i) {
        Node *leaf = node_select_leaf(root);
        if (!node_is_terminal(leaf)) {
            MoveList ml;
            generate_moves(&leaf->gs.board, leaf->gs.current_player == WHITE, &ml);
            node_expand(leaf, &ml);
            GameResult r = playout(leaf->gs);
            node_backprop(leaf, r);
        }
    }

    // Each iteration: fill cur buffer with leaf_count leaves * PLAYOUTS_PER_NODE copies,
    // launch async H2D memcpy + kernel on stream[cur], while simultaneously
    // retrieving results (async D2H) and backprop for prev buffer.
    for (;;) {
        double elapsed = ((double)clock() - t0) / CLOCKS_PER_SEC;
        if (elapsed >= 0.98 * timeout)
            break;

        size_t leaf_count = 0;
        for (; leaf_count < BATCH; ++leaf_count) {
            elapsed = ((double)clock() - t0) / CLOCKS_PER_SEC;
            if (elapsed >= 0.98 * timeout)
                break;

            Node *leaf = node_select_leaf(root, PLAYOUTS_PER_NODE);

            if (!node_is_terminal(leaf)) {
                MoveList ml;
                generate_moves(&leaf->gs.board, leaf->gs.current_player == WHITE, &ml);
                node_expand(leaf, &ml);
            }

            // schedule PLAYOUTS_PER_NODE playouts for this leaf
            for (size_t p = 0; p < PLAYOUTS_PER_NODE; ++p) {
                size_t idx = leaf_count * PLAYOUTS_PER_NODE + p;
                leaf_batch_host[cur][idx] = leaf;
                h_states[cur][idx] = leaf->gs;
            }
        }

        size_t cur_total = leaf_count * PLAYOUTS_PER_NODE;

        // Async copy H->D for cur buffer and launch kernel on stream[cur]
        CUDA_CHECK(cudaMemcpyAsync(d_states[cur], h_states[cur],
                                   cur_total * sizeof(GameState),
                                   cudaMemcpyHostToDevice, streams[cur]));

        int block = 32;
        int grid = (cur_total + block - 1) / block;
        gpu_playout_kernel<<<grid, block, 0, streams[cur]>>>(d_states[cur], d_results[cur], (int)cur_total);
        CUDA_CHECK(cudaGetLastError());

        // Issue D2H async copy for previous buffer (if any) and then synchronize that stream and backprop
        if (prev != -1) {
            CUDA_CHECK(cudaMemcpyAsync(h_results[prev], d_results[prev],
                                       prev_total * sizeof(GameResult),
                                       cudaMemcpyDeviceToHost, streams[prev]));

            CUDA_CHECK(cudaStreamSynchronize(streams[prev]));

            for (size_t i = 0; i < prev_total; ++i) {
                node_backprop(leaf_batch_host[prev][i], h_results[prev][i]);
            }
            total_playouts += prev_total;
        }

        // swap buffers
        prev = cur;
        prev_total = cur_total;
        cur ^= 1;

        iterations++;
    }

    // After main loop, we may have one in-flight buffer (prev). Wait and process it.
    if (prev != -1 && prev_total > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_results[prev], d_results[prev],
                                   prev_total * sizeof(GameResult),
                                   cudaMemcpyDeviceToHost, streams[prev]));
        CUDA_CHECK(cudaStreamSynchronize(streams[prev]));
        for (size_t i = 0; i < prev_total; ++i) {
            node_backprop(leaf_batch_host[prev][i], h_results[prev][i]);
        }
        total_playouts += prev_total;
    }

    for (int i = 0; i < root->children.count; ++i) {
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

cleanup:
    if (streams[0]) cudaStreamDestroy(streams[0]);
    if (streams[1]) cudaStreamDestroy(streams[1]);

    if (d_states[0]) cudaFree(d_states[0]);
    if (d_states[1]) cudaFree(d_states[1]);
    if (d_results[0]) cudaFree(d_results[0]);
    if (d_results[1]) cudaFree(d_results[1]);

    if (h_states[0]) cudaFreeHost(h_states[0]);
    if (h_states[1]) cudaFreeHost(h_states[1]);
    if (h_results[0]) cudaFreeHost(h_results[0]);
    if (h_results[1]) cudaFreeHost(h_results[1]);

    if (leaf_batch_host[0]) free(leaf_batch_host[0]);
    if (leaf_batch_host[1]) free(leaf_batch_host[1]);

    if (root) node_free(root);

    return best_move;
}


#endif /* MCTS_GPU_H */
