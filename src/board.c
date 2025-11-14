#include "board.h"
#include "helpers.h"
#include "movegen.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef FORMATTING
#define FORM_END "\e[m"
#define FORM_FADE "\e[2m"
#define FORM_UNDER "\e[4m"
#else
#define FORM_END ""
#define FORM_FADE ""
#define FORM_UNDER ""
#endif /* FORMATTING */

#define BOT_ROW 0x00041041
#define TOP_ROW 0x82000820

const int idx_to_board[32] = {
    11, 5,  31, 25, 10, 4,  30, 24, 3,  29, 23, 17, 2,  28, 22, 16,
    27, 21, 15, 9,  26, 20, 14, 8,  19, 13, 7,  1,  18, 12, 6,  0,
};

void init_board(Board *board) {
    board->kings = 0;
    board->white = 0xE3820C38; // top 3 rows
    board->black = 0x041C71C3; // bot 3 rows
}

void print_board(const Board *board, const MoveList *mlist,
                 const Move *last_move) {
    printf("\n  a b c d e f g h\n");
    for (int y = 0; y < 8; ++y) {
        printf("%d ", 8 - y);
        for (int x = 0; x < 8; ++x) {
            char c = '.';
            int idx = idx_to_board[y * 4 + x / 2];
            if ((x + y) % 2 == 0) {
                printf(FORM_FADE ". " FORM_END);
                continue;
            }
            u32 mask = 1 << idx;
            if (board->white & mask) {
                c = board->kings & mask ? 'W' : 'w';
            } else if (board->black & mask) {
                c = board->kings & mask ? 'B' : 'b';
            }
            mask <<= 1;
            int valid = 0;
            for (int i = 0; i < mlist->count; ++i) {
                if (mlist->moves[i].path[0] == idx) {
                    valid = 1;
                    break;
                }
            }
            if (valid)
                printf("%c ", c);
            else if (last_move->path_len > 1 && last_move->path[0] == idx ||
                     last_move->path[1] == idx)
                printf(FORM_FADE FORM_UNDER "%c" FORM_END " ", c);
            else
                printf(FORM_FADE "%c " FORM_END, c);
        }
        printf("%d\n", 8 - y);
    }
    printf("  a b c d e f g h\n");
}

void apply_move(Board *board, const Move *m) {
    u32 from_mask = 1u << m->path[0];
    u32 to_mask = 1u << m->path[m->path_len - 1];

    int is_white = (board->white & from_mask) != 0;
    int is_king = (board->kings & from_mask) != 0;

    u32 clear_mask = from_mask | m->captured;
    board->white &= ~clear_mask;
    board->black &= ~clear_mask;
    board->kings &= ~clear_mask;

    // place on destination
    if (is_white)
        board->white |= to_mask;
    else
        board->black |= to_mask;

    // promotion: if not already a king and reaches last row
    if (is_king || (is_white && (BOT_ROW & to_mask)) ||
        (!is_white && (TOP_ROW & to_mask)))
        board->kings |= to_mask;
}

static inline void index_to_xy(int idx, int *x, int *y) {
    *y = idx / 4;
    *x = 2 * (idx % 4) + ((*y + 1) % 2);
}
static inline int sgn(int v) { return (v > 0) - (v < 0); }

// compute captured squares by inspecting segments of the path and looking for
// opponent pieces
// TODO: simplify!, probably remove this and update capture mask as the move is created
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

