#include "movegen.h"
#include "board.h"
#include "move.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void append_move(MoveList *l, Move m) { l->moves[l->count++] = m; }

// for sorting movelist alphabetically
int pstrcmp(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

void print_movelist(const MoveList *l) {
    char **moves_str = malloc(l->count * sizeof(char *));
    if (!moves_str)
        return;

    for (int i = 0; i < l->count; ++i) {
        moves_str[i] = malloc(32);
        move_to_str(&l->moves[i], moves_str[i]);
    }

    qsort(moves_str, l->count, sizeof(moves_str[0]), pstrcmp);

    for (int i = 0; i < l->count; ++i) {
        printf("%s", moves_str[i]);
        if (i + 1 < l->count)
            printf(", ");
    }
    printf("\n");

    for (int i = 0; i < l->count; ++i)
        free(moves_str[i]);
    free(moves_str);
}

int is_valid_move(const Move *m, const MoveList *l) {
    for (int i = 0; i < l->count; ++i) {
        if (m->path_len != l->moves[i].path_len)
            continue;
        int match = 1;
        for (int j = 0; j < m->path_len; ++j) {
            if (m->path[j] != l->moves[i].path[j]) {
                match = 0;
                break;
            }
        }
        if (match)
            return 1;
    }

    return 0;
}

static void force_captures(MoveList *l) {
    int capture_len = 0;
    for (int i = 0; i < l->count; ++i) {
        if (is_capture(&l->moves[i])) {
            int len = l->moves[i].path_len;
            capture_len = len > capture_len ? len : capture_len;
            break;
        }
    }
    if (!capture_len)
        return;

    // retain only longest captures
    int cnt = 0;
    for (int i = 0; i < l->count; ++i) {
        if (is_capture(&l->moves[i]) && l->moves[i].path_len == capture_len) {
            l->moves[cnt++] = l->moves[i];
        }
    }
    l->count = cnt;
}

void generate_single(const Board *b, int is_white, MoveList *out) {
    u32 own = is_white ? b->white : b->black;
    u32 ene = is_white ? b->black : b->white;
    u32 occ = own | ene;
    u32 free = ~occ;

    u32 can_move_down = 0xFFFFFFF0;  // + 4
    u32 can_move_right = 0x07070707; // + 5
    u32 can_move_left = 0xE0E0E0E0;  // + 3

    Move m;
    for (int idx = 0; idx < 32; ++idx) {
        u32 idxm = 1u << idx;
        if (!(own & idxm))
            continue;

        if (can_move_down & idxm) {
            int mv = idx + 4;
            u32 p = idxm << 4;
            if (free & p) {
                simple_move(idx, mv, &m);
                append_move(out, m);
            } else if (ene & p) {
                if (can_move_right & p && free & (p << 5)) {
                    mv += 5;
                    simple_move(idx, mv, &m);
                    append_move(out, m);
                } else if (can_move_left & p && free & (p << 3)) {
                    mv += 3;
                    simple_move(idx, mv, &m);
                    append_move(out, m);
                }
            }
        }
        if (can_move_right & idxm) {
            int mv = idx + 5;
            u32 p = idxm << 5;
            if (free & p) {
                simple_move(idx, mv, &m);
                append_move(out, m);
            } else if (ene & p) {
                if (can_move_down & p && free & (p << 4)) {
                    mv += 4;
                    simple_move(idx, mv, &m);
                    append_move(out, m);
                }
            }
        }
        if (can_move_left & idxm) {
            int mv = idx + 3;
            u32 p = idxm << 3;
            if (free & p) {
                simple_move(idx, mv, &m);
                append_move(out, m);
            } else if (ene & p) {
                if (can_move_down & p && free & (p << 4)) {
                    mv += 4;
                    simple_move(idx, mv, &m);
                    append_move(out, m);
                }
            }
        }
    }
}

void flip_moves(MoveList *l) {
    for (int i = 0; i < l->count; ++i)
        flip_move(&l->moves[i]);
}

void generate_moves(const Board *b, int is_white, MoveList *out) {
    out->count = 0;

    if (!is_white) {
        Board board = *b;
        flip_perspective(&board);
        generate_single(&board, is_white, out);
        flip_moves(out);
    } else {
        generate_single(b, is_white, out);
    }

    force_captures(out);
}
