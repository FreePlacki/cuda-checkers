#include "move.h"
#include "board.h"
#include "helpers.h"
#include <ctype.h>

static inline int pos_to_index(char file, char rank) {
    if (!((file >= 'a' && file <= 'h') || (rank >= '1' && rank <= '8')))
        return -1;
    int x = file - 'a';
    int y = 8 - (rank - '0');
    if ((x + y) % 2 == 0)
        return -1; // light square
    return idx_to_board[y * 4 + (x / 2)];
}

void simple_move(u8 from, u8 to, Move *out) {
    out->path_len = 2;
    out->path[0] = from;
    out->path[1] = to;
    out->captured = 0;
}

int is_capture(const Move *move) { return move->captured != 0; }

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
        int fd = (dst + 32 - src) % 32;

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
            int mid = (src + step) % 32;
            out->captured |= (1u << mid);
        }
    }

    return 1;
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
