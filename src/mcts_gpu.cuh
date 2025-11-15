#ifndef MCTS_GPU_H
#define MCTS_GPU_H

#include "assert.h"
#include "board.cuh"
#include "gamestate.cuh"
#include "move.cuh"
#include "movegen.cuh"
#include "time.h"

// https://en.wikipedia.org/wiki/Xorshift
__device__ inline uint32_t xorshift32(u32 *s) {
    u32 x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

__device__ GameResult playout_device(GameState *gs, u32 rand_seed) {
    MoveList mlist;
    GameState state;

    memcpy(&state, gs, sizeof(GameState));

    for (;;) {
        generate_moves(&state.board, state.current_player == WHITE, &mlist);
        if (mlist.count == 0)
            return state.current_player == WHITE ? BLACK_WON : WHITE_WON;

        u32 r = xorshift32(&rand_seed);
        Move m = mlist.moves[r % mlist.count];
        apply_move(&state.board, &m, 1);
        next_turn(&state, is_capture(&m));
        GameResult res = game_result(&state);
        if (res != PENDING)
            return res;
    }
}

__global__ void gpu_playout_kernel(GameState *states, GameResult *results,
                                   int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    GameState s = states[id];
    u32 seed = 0x9E3779B9u ^ id;

    results[id] = playout_device(&s, seed);
}

int playout_gpu(const GameState *gs, int playouts) {
    GameState *d_states = nullptr;
    GameResult *d_results = nullptr;
    GameResult *h_results = new GameResult[playouts];

    // Allocate device memory
    cudaMalloc(&d_states, playouts * sizeof(GameState));
    cudaMalloc(&d_results, playouts * sizeof(GameResult));

    // Create a host array with 'playouts' copies of gs
    GameState *h_states = new GameState[playouts];
    for (int i = 0; i < playouts; ++i)
        h_states[i] = *gs;

    cudaMemcpy(d_states, h_states, playouts * sizeof(GameState),
               cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (playouts + block - 1) / block;
    gpu_playout_kernel<<<grid, block>>>(d_states, d_results, playouts);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, playouts * sizeof(GameResult),
               cudaMemcpyDeviceToHost);

    // Score: white win = +1, black win = -1
    int score = 0;
    for (int i = 0; i < playouts; ++i) {
        if (h_results[i] == WHITE_WON)
            score++;
        else if (h_results[i] == BLACK_WON)
            score--;
    }

    delete[] h_results;
    delete[] h_states;
    cudaFree(d_states);
    cudaFree(d_results);

    return score;
}

Move choose_move_flat_gpu(const GameState *gs, const MoveList *l) {
    if (l->count == 1)
        return l->moves[0];

    const int playouts = 10000;
    double t0 = now();

    int best_score = -playouts;
    int best_idx = 0;

    for (int i = 0; i < l->count; ++i) {

        GameState next = *gs;
        apply_move(&next.board, &l->moves[i], 1);
        next_turn(&next, is_capture(&l->moves[i]));

        int score = playout_gpu(&next, playouts);

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    printf("valuation: %.4f\nplayouts: %d\ntook: %lf s\n",
           (double)best_score / playouts, playouts, now() - t0);

    return l->moves[best_idx];
}

#endif /* MCTS_GPU_H */
