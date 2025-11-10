#include "board.h"
#include <stdio.h>
#include <stdlib.h>

void init_board(Board *board) {
    board->kings = 0u;
    board->white = 0x00000FFFu;
    board->black = 0xFFF00000u;
}

void print_board(const Board *board) {
    printf("\n  a b c d e f g h\n");
    int mask = 1u;
    for (int y = 0; y < 8; ++y) {
        printf("%d ", 8 - y);
        for (int x = 0; x < 8; ++x) {
            char c = '.';
            if ((x + y) % 2) {
                if (board->white & mask) {
                    c = board->kings & mask ? 'W' : 'w';
                } else if (board->black & mask) {
                    c = board->kings & mask ? 'B' : 'b';
                }
                mask <<= 1;
            }
            printf("%c ", c);
        }
        printf("%d\n", 8 - y);
    }
    printf("  a b c d e f g h\n");
}

void apply_move(Board *board, const Move *m) {
    uint32_t captures = 0;
    move_compute_captures(board, m, &captures);

    uint32_t from_mask = 1u << m->path[0];
    uint32_t to_mask = 1u << m->path[m->path_len - 1];
    uint32_t clear_mask = from_mask | captures;

    int is_white = (board->white & from_mask) != 0;
    int is_king = (board->kings & from_mask) != 0;

    // clear from and captured
    board->white &= ~clear_mask;
    board->black &= ~clear_mask;
    board->kings &= ~clear_mask;

    // place on destination
    if (is_white)
        board->white |= to_mask;
    else
        board->black |= to_mask;

    // promotion: if not already a king and reaches last row
    int to_row = (int)(m->path[m->path_len - 1]) / 4;
    if (is_king || (is_white && to_row == 7) || (!is_white && to_row == 0))
        board->kings |= to_mask;
}

static inline void index_to_xy(int idx, int *x, int *y) {
    *y = idx / 4;
    *x = 2 * (idx % 4) + ((*y + 1) % 2);
}
static inline int sgn(int v) { return (v > 0) - (v < 0); }

// compute captured squares by inspecting segments of the path and looking for
// opponent pieces
int move_compute_captures(const Board *b, const Move *m,
                          uint32_t *captures_out) {
    if (!b || !m || !captures_out)
        return 0;
    uint32_t caps = 0;
    if (m->path_len < 2)
        return 0;

    // determine color of moving piece from starting square
    uint32_t from_mask = 1u << m->path[0];
    int is_white = (b->white & from_mask) != 0;
    int is_king = (b->kings & from_mask) != 0;

    // iterate segments
    for (int s = 0; s < m->path_len - 1; ++s) {
        int a = m->path[s];
        int bidx = m->path[s + 1];

        int ax, ay, bx, by;
        index_to_xy(a, &ax, &ay);
        index_to_xy(bidx, &bx, &by);

        int dx = bx - ax;
        int dy = by - ay;
        if (dx == 0 || dy == 0)
            return 0; // must be diagonal
        if (abs(dx) != abs(dy))
            return 0; // must move along diagonal

        int stepx = sgn(dx);
        int stepy = sgn(dy);
        int steps = abs(dx);

        if (!is_king && steps != 2) {
            // men must jump exactly two squares when capturing
            return 0;
        }

        // search for captured piece(s) between a and b (exclusive)
        int found = 0;
        int cx = ax + stepx;
        int cy = ay + stepy;
        for (int t = 1; t < steps; ++t, cx += stepx, cy += stepy) {
            int mid_idx = cy * 4 + (cx / 2);
            uint32_t mid_mask = 1u << mid_idx;
            // if an opponent occupies this square, mark it captured
            if (is_white) {
                if ((b->black & mid_mask) != 0) {
                    caps |= mid_mask;
                    found++;
                }
            } else {
                if ((b->white & mid_mask) != 0) {
                    caps |= mid_mask;
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

    *captures_out = caps;
    return 1;
}
