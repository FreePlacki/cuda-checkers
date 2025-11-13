#include "board.h"
#include "helpers.h"
#include "movegen.h"
#include <stdio.h>
#include <stdlib.h>

void init_board(Board *board) {
    board->kings = 0u;
    board->white = 0x00000FFFu;
    board->black = 0xFFF00000u;
}

void print_board(const Board *board, const MoveList *mlist,
                 const Move *last_move) {
    printf("\n  a b c d e f g h\n");
    int mask = 1u;
    for (int y = 0; y < 8; ++y) {
        printf("%d ", 8 - y);
        for (int x = 0; x < 8; ++x) {
            char c = '.';
            int idx = y * 4 + x / 2;
            if ((x + y) % 2 == 0) {
                printf("\e[2m.\e[m ");
                continue;
            }
            if (board->white & mask) {
                c = board->kings & mask ? 'W' : 'w';
            } else if (board->black & mask) {
                c = board->kings & mask ? 'B' : 'b';
            }
            mask <<= 1;
            int valid = 0;
            for (int i = 0; i < mlist->count; ++i) {
                if (mlist->moves[i].path[0] == y * 4 + x / 2) {
                    valid = 1;
                    break;
                }
            }
            if (valid)
                printf("%c ", c);
            else if (last_move->path_len > 1 && last_move->path[0] == idx ||
                     last_move->path[1] == idx)
                printf("\e[4m\e[2m%c\e[m\e[m ", c);
            else
                printf("\e[2m%c\e[m ", c);
        }
        printf("%d\n", 8 - y);
    }
    printf("  a b c d e f g h\n");
}

void apply_move(Board *board, const Move *m) {
    u32 captures = 0u;
    move_compute_captures(board, m, &captures);

    u32 from_mask = 1u << m->path[0];
    u32 to_mask = 1u << m->path[m->path_len - 1];

    int is_white = (board->white & from_mask) != 0;
    int is_king = (board->kings & from_mask) != 0;

    u32 clear_mask = from_mask | captures;
    board->white &= ~clear_mask;
    board->black &= ~clear_mask;
    board->kings &= ~clear_mask;

    // place on destination
    if (is_white)
        board->white |= to_mask;
    else
        board->black |= to_mask;

    // promotion: if not already a king and reaches last row
    int to_row_white = 0xF0000000;
    int to_row_black = 0x0000000F;
    if (is_king || (is_white && (to_row_white & to_mask)) ||
        (!is_white && (to_row_black & to_mask)))
        board->kings |= to_mask;
}

static inline void index_to_xy(int idx, int *x, int *y) {
    *y = idx / 4;
    *x = 2 * (idx % 4) + ((*y + 1) % 2);
}
static inline int sgn(int v) { return (v > 0) - (v < 0); }

// compute captured squares by inspecting segments of the path and looking for
// opponent pieces
// TODO: simplify!
int move_compute_captures(const Board *b, const Move *m, u32 *caps) {
    // determine color of moving piece from starting square
    u32 from_mask = 1u << m->path[0];
    int is_white = (b->white & from_mask) != 0;

    // iterate segments
    for (int s = 0; s < m->path_len - 1; ++s) {
        int a = m->path[s];
        int bidx = m->path[s + 1];

        int ax, ay, bx, by;
        index_to_xy(a, &ax, &ay);
        index_to_xy(bidx, &bx, &by);

        int dx = bx - ax;
        int dy = by - ay;

        int stepx = sgn(dx);
        int stepy = sgn(dy);
        int steps = abs(dx);

        // search for captured piece(s) between a and b (exclusive)
        int found = 0;
        int cx = ax + stepx;
        int cy = ay + stepy;
        for (int t = 1; t < steps; ++t, cx += stepx, cy += stepy) {
            int mid_idx = cy * 4 + (cx / 2);
            u32 mid_mask = 1u << mid_idx;
            // if an opponent occupies this square, mark it captured
            if (is_white) {
                if ((b->black & mid_mask) != 0) {
                    *caps |= mid_mask;
                    found++;
                }
            } else {
                if ((b->white & mid_mask) != 0) {
                    *caps |= mid_mask;
                    found++;
                }
            }
        }

        if (!found) {
            // no captured piece in this segment -> illegal capture
            return 0;
        }
        // for standard rules found should be exactly 1 per segment
    }

    return 1;
}

// on CUDA use __brev
// (https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html#_CPPv46__brevj)
static u32 reverse_bits(u32 x) {
    x = (x & 0xAAAAAAAA) >> 1 | (x & 0x55555555) << 1;
    x = (x & 0xCCCCCCCC) >> 2 | (x & 0x33333333) << 2;
    x = (x & 0xF0F0F0F0) >> 4 | (x & 0x0F0F0F0F) << 4;
    x = (x & 0xFF00FF00) >> 8 | (x & 0x00FF00FF) << 8;
    return (x << 16) | (x >> 16);
}

Board flip_perspective(const Board *b) {
    Board board;
    board.black = reverse_bits(b->black);
    board.white = reverse_bits(b->white);
    board.kings = reverse_bits(b->kings);

    return board;
}
