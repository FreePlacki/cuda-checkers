#ifndef MOVE_H
#define MOVE_H

#include "board.cuh"
#include "helpers.h"
#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define MOVE_STR_MAX 36

// The move is represended by a mask where a bit is set if it's either:
// - the piece's starting position
// - the piece's ending position
// - the position of a captured piece
// To apply the move simply xor it with the board.
// This introduces a problem if the piece's starting pos = it's ending pos,
// thus we have to check that the number of allied pieces remains constant
// when applying the move.
typedef u32 Move;

__host__ __device__ __forceinline__ int popcnt(u32 x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#elif defined(_MSC_VER)
    return __popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

// counts the number of consecutive 0s (from least significant)
__host__ __device__ __forceinline__ int ctz32(u32 x) {
#if defined(__CUDA_ARCH__)
    return __ffs(x) - 1;
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, x);
    return idx;
#else
    return __builtin_ctz(x);
#endif
}

static inline int pos_to_index(char file, char rank) {
    if (!((file >= 'a' && file <= 'h') || (rank >= '1' && rank <= '8')))
        return -1;
    int x = file - 'a';
    int y = 8 - (rank - '0');
    if ((x + y) % 2 == 0)
        return -1; // light square
    return idx_to_board[y * 4 + (x / 2)];
}

__host__ __device__ void simple_move(u8 from, u8 to, Move *out) {
    *out = (1u << from) | (1u << to);
}

__host__ __device__ int is_capture(Move move) { return popcnt(move) > 2; }

// returns a mask with one bit set - the starting move position
__host__ __device__ u32 move_start(const Board *b, Move move, int is_white) {
    u32 own = is_white ? b->white : b->black;
    return own & move;
}

// returns a mask with one bit set - the ending move position
__host__ __device__ u32 move_end(const Board *b, Move move, int is_white) {
    u32 ene = is_white ? b->black : b->white;

    u32 st = move_start(b, move, is_white);
    u32 end = ~st & ~ene & move;

    // the case where start_pos == end_pos
    if (end == 0)
        end = st;

    return end;
}

int parse_move(const char *str, Move *out) {
    int path_len = 0;
    u8 path[10];

    int expect_square = 1;
    while (*str) {
        while (*str && isspace((unsigned char)*str))
            str++;
        if (!*str)
            break;

        if (expect_square) {
            if (!isalpha((unsigned char)str[0]) ||
                !isdigit((unsigned char)str[1]))
                return 0;
            int idx = pos_to_index(str[0], str[1]);
            if (idx < 0)
                return 0;
            if (path_len >= 10)
                return 0;
            path[path_len++] = (u8)idx;
            str += 2;
            expect_square = 0;
        } else {
            if (*str != '-' && *str != ':')
                return 0;
            str++;
            expect_square = 1;
        }
    }

    if (path_len < 2)
        return 0;

    *out = 0;
    *out |= 1u << path[0];
    *out |= 1u << path[path_len - 1];

    // reconstruct captured
    for (int i = 0; i < path_len - 1; ++i) {
        int src = path[i];
        int dst = path[i + 1];
        int fd = (dst - src) & 31;

        int step;
        int is_capture = 0;

        switch (fd) {
        case 2:
            step = 1;
            is_capture = 1;
            break;
        case 14:
            step = 7;
            is_capture = 1;
            break;
        case 30:
            step = -1;
            is_capture = 1;
            break;
        case 18:
            step = -7;
            is_capture = 1;
            break;
        }

        if (is_capture) {
            int mid = (src + step) & 31;
            *out |= (1u << mid);
        }
    }

    return 1;
}

__host__ __device__ void apply_move(Board *board, Move move, int is_white,
                                    int with_promotion) {
    u32 *own = is_white ? &board->white : &board->black;
    u32 *ene = is_white ? &board->black : &board->white;

    u32 start = *own & move;
    u32 end = ~(*own) & ~(*ene) & move;
    u32 capt = move & (~start) & (~end);
    int is_king = board->kings & start;

    // the case where start_pos == end_pos
    if (end == 0)
        end = start;

    *own ^= start;
    *own |= end;
    *ene ^= capt;
    board->kings &= ~capt;
    board->kings &= ~start;

    // promotion: if not already a king and reaches last row
    int reached_prom_row =
        (is_white && (BOT_ROW & end)) || (!is_white && (TOP_ROW & end));
    if (is_king || (with_promotion && reached_prom_row))
        board->kings |= end;
}

static void index_to_algebraic(u8 idx, char out[2]) {
    u8 y = 8 - idx / 4;
    u8 x = 2 * (idx % 4) + (y % 2 == 0);
    out[0] = 'a' + x;
    out[1] = '0' + y;
}

static int index_to_board(u8 idx) {
    for (int i = 0; i < 32; ++i) {
        if (idx_to_board[i] == idx)
            return i;
    }
    return -1;
}

void move_to_str(const Board *board, Move move, int is_white, char *out) {
    int j = 0;
    u32 st = move_start(board, move, is_white);
    u32 en = move_end(board, move, is_white);

    u8 st_idx = ctz32(st);
    u8 en_idx = ctz32(en);
    char s[2];
    index_to_algebraic(index_to_board(st_idx), s);
    out[j++] = s[0];
    out[j++] = s[1];

    if (!is_capture(move)) {
        out[j++] = '-';
        index_to_algebraic(index_to_board(en_idx), s);
        out[j++] = s[0];
        out[j++] = s[1];
        out[j] = 0;
        return;
    }

    u8 pos = st_idx;
    do {
        u32 mask = 1u << pos;
        u32 pos_m;
        int found = 0;

        if (!found && (CAN_DL & mask)) {
            u32 cap = rotr(mask, 1);
            u32 land = rotr(cap, 1);
            if (cap & move) {
                move &= ~cap;
                pos_m = land;
                found = 1;
            }
        }
        if (!found && (CAN_DR & mask)) {
            u32 cap = rotr(mask, 7);
            u32 land = rotr(cap, 7);
            if (cap & move) {
                move &= ~cap;
                pos_m = land;
                found = 1;
            }
        }
        if (!found && (CAN_UR & mask)) {
            u32 cap = rotl(mask, 1);
            u32 land = rotl(cap, 1);
            if (cap & move) {
                move &= ~cap;
                pos_m = land;
                found = 1;
            }
        }
        if (!found && (CAN_UL & mask)) {
            u32 cap = rotl(mask, 7);
            u32 land = rotl(cap, 7);
            if (cap & move) {
                move &= ~cap;
                pos_m = land;
                found = 1;
            }
        }
        if (!found)
            break;
        pos = ctz32(pos_m);

        out[j++] = ':';
        index_to_algebraic(index_to_board(pos), s);
        out[j++] = s[0];
        out[j++] = s[1];
    } while (pos != en_idx);

    out[j] = 0;
}

#endif /* MOVE_H */
