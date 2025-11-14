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

void apply_move(Board *board, const Move *m, int with_promotion) {
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
    if (with_promotion && (is_king || (is_white && (BOT_ROW & to_mask)) ||
                           (!is_white && (TOP_ROW & to_mask))))
        board->kings |= to_mask;
}
