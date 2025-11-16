#ifndef MOVE_H
#define MOVE_H

#include "board.cuh"
#include <stdint.h>

#define MOVE_STR_MAX 36

// TODO: Optimize the size
// idea:
// u32 path  -- bitmask of piece intermediate positions and captures
// u8 begin  -- initial piece position
// u8 end    -- final piece position
// total size: 8 bytes
// cons: we cannot implement flying king rules with this approach
typedef struct {
    u8 path[10];  // sequence of visited squares: from, ..., to
    u8 path_len;  // number of squares in path (>=2)
    u32 captured; // mask for pieces captured by this move
} Move;

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
    out->path_len = 2;
    out->path[0] = from;
    out->path[1] = to;
    out->captured = 0;
}

__host__ __device__ int is_capture(const Move *move) {
    return move->captured != 0;
}

int parse_move(const char *str, Move *out) {
    out->path_len = 0;
    out->captured = 0;

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
            if (out->path_len >= 10)
                return 0;
            out->path[out->path_len++] = (uint8_t)idx;
            str += 2;
            expect_square = 0;
        } else {
            if (*str != '-' && *str != ':')
                return 0;
            str++;
            expect_square = 1;
        }
    }

    if (out->path_len < 2)
        return 0;

    // reconstruct captured
    for (int i = 0; i < out->path_len - 1; ++i) {
        int src = out->path[i];
        int dst = out->path[i + 1];
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
            out->captured |= (1u << mid);
        }
    }

    return 1;
}

__host__ __device__ void apply_move(Board *board, const Move *m,
                                    int with_promotion) {
    u32 from_mask = 1u << m->path[0];
    u32 to_mask = 1u << m->path[m->path_len - 1];

    int is_white = (board->white & from_mask) != 0;
    int is_king = (board->kings & from_mask) != 0;

    u32 clear_mask = from_mask | m->captured;
    board->white &= ~clear_mask;
    board->black &= ~clear_mask;
    board->kings &= ~clear_mask;

    if (is_white)
        board->white |= to_mask;
    else
        board->black |= to_mask;

    // promotion: if not already a king and reaches last row
    int reached_prom_row = (is_white && (BOT_ROW & to_mask)) ||
        (!is_white && (TOP_ROW & to_mask));
    if (is_king || (with_promotion && reached_prom_row))
        board->kings |= to_mask;
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

void move_to_str(const Move *move, char *out) {
    int j = 0;
    for (int i = 0; i < move->path_len; ++i) {
        u8 idx = move->path[i];
        char s[2];
        index_to_algebraic(index_to_board(idx), s);

        out[j++] = s[0];
        out[j++] = s[1];
        if (i + 1 < move->path_len)
            out[j++] = is_capture(move) ? ':' : '-';
    }
    out[j] = 0;
}

#endif /* MOVE_H */
