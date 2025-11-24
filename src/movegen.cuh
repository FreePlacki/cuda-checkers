#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "assert.h"
#include "board.cuh"
#include "helpers.h"
#include "move.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOVELIST_SIZE 32
typedef struct MoveList {
    Move moves[MOVELIST_SIZE];
    u8 count;
} MoveList;

typedef struct {
    Move moves[4];
    u8 count;
} SmallMoveList;

typedef struct {
    Move *moves;
    u8 *count;
} MoveListView;

__host__ __device__ MoveListView view_from_movelist(MoveList *l) {
    return (MoveListView){l->moves, &l->count};
}

__host__ __device__ MoveListView view_from_small(SmallMoveList *s) {
    return (MoveListView){s->moves, &s->count};
}

__host__ __device__ void append_move(MoveList *l, Move m) {
    if (l->count == MOVELIST_SIZE)
        return;
    l->moves[l->count++] = m;
}

__host__ __device__ void view_append_move(MoveListView *l, Move m) {
    if (*l->count == MOVELIST_SIZE)
        return;
    l->moves[(*l->count)++] = m;
}

// for sorting movelist alphabetically
static int pstrcmp(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

void print_movelist(const Board *board, const MoveList *l, int is_white) {
    if (l->count == 0)
        return;
    char **moves_str = (char **)malloc(l->count * sizeof(char *));
    if (!moves_str)
        return;

    for (int i = 0; i < l->count; ++i) {
        moves_str[i] = (char *)malloc(sizeof(char *));
        move_to_str(board, l->moves[i], is_white, moves_str[i]);
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

int is_valid_move(Move m, const MoveList *l) {
    for (int i = 0; i < l->count; ++i) {
        if (m == l->moves[i])
            return 1;
    }

    return 0;
}

__host__ __device__ void force_captures(MoveList *l) {
    int capture_len = 0;
    for (int i = 0; i < l->count; ++i) {
        if (is_capture(l->moves[i])) {
            int len = popcnt(l->moves[i]);
            capture_len = len > capture_len ? len : capture_len;
        }
    }
    if (capture_len == 0)
        return;

    int cnt = 0;
    for (int i = 0; i < l->count; ++i) {
        if (is_capture(l->moves[i])) {
            l->moves[cnt++] = l->moves[i];
        }
    }
    l->count = cnt;
}

__host__ __device__ void gen_shift_moves(u32 own, u32 free, int shift,
                                         MoveListView *out) {
    u32 shifted = shift > 0 ? rotl(own, shift) : rotr(own, -shift);

    u32 moves = shifted & free;

    // iterating 1s (from least significant)
    while (moves) {
        u32 dst = moves & -moves;
        moves ^= dst;

        int dst_idx = ctz32(dst);
        int src_idx = (dst_idx - shift) & 31;

        Move m;
        simple_move(src_idx, dst_idx, &m);
        view_append_move(out, m);
    }
}

__host__ __device__ int gen_shift_capture(u32 ownp, u32 ene, u32 free, int step,
                                          MoveListView *out) {
    u32 nei = step > 0 ? rotl(ownp, step) : rotr(ownp, -step);
    nei &= ene;
    u32 dst = (step > 0) ? rotl(nei, step) : rotr(nei, -step);
    dst &= free;

    int cap_found = dst != 0;
    while (dst) {
        u32 dst_b = dst & -dst;
        dst ^= dst_b;

        int dst_idx = ctz32(dst_b);
        int cap_idx = (dst_idx - step) & 31;
        int src_idx = (cap_idx - step) & 31;

        Move m;
        simple_move(src_idx, dst_idx, &m);
        m |= 1u << cap_idx;
        view_append_move(out, m);
    }

    return cap_found;
}

__host__ __device__ int generate_single(const Board *b, int is_white,
                                        MoveListView *out, int mask) {
    *out->count = 0;

    const u32 own = is_white ? b->white : b->black;
    const u32 ene = is_white ? b->black : b->white;
    const u32 occ = own | ene;
    const u32 free = ~occ;

    const u32 kings = b->kings & own & mask;
    const u32 men = own & ~kings & mask;

    int capt = 0;
    if (is_white) {
        capt |= gen_shift_capture(men & CAN_DL, ene & CAN_DL, free, -1, out);
        capt |= gen_shift_capture(men & CAN_DR, ene & CAN_DR, free, -7, out);

        if (!capt) {
            gen_shift_moves(men & CAN_DL, free, -1, out);
            gen_shift_moves(men & CAN_DR, free, -7, out);
        }
    } else {
        capt |= gen_shift_capture(men & CAN_UL, ene & CAN_UL, free, +7, out);
        capt |= gen_shift_capture(men & CAN_UR, ene & CAN_UR, free, +1, out);

        if (!capt) {
            gen_shift_moves(men & CAN_UR, free, +1, out);
            gen_shift_moves(men & CAN_UL, free, +7, out);
        }
    }

    capt |= gen_shift_capture(kings & CAN_UR, ene & CAN_UR, free, +1, out);
    capt |= gen_shift_capture(kings & CAN_UL, ene & CAN_UL, free, +7, out);
    capt |= gen_shift_capture(kings & CAN_DR, ene & CAN_DR, free, -7, out);
    capt |= gen_shift_capture(kings & CAN_DL, ene & CAN_DL, free, -1, out);

    if (!capt) {
        gen_shift_moves(kings & CAN_UR, free, +1, out);
        gen_shift_moves(kings & CAN_UL, free, +7, out);
        gen_shift_moves(kings & CAN_DR, free, -7, out);
        gen_shift_moves(kings & CAN_DL, free, -1, out);
    }

    return capt;
}

typedef struct {
    Move m;
    Board board;
} StackNode;

#define MOVES_STCK_MAX 32
typedef struct {
    StackNode nodes[MOVES_STCK_MAX];
    int sp;
} MoveS;

__host__ __device__ __forceinline__ int push(StackNode n, MoveS *s) {
    if (s->sp >= MOVES_STCK_MAX)
        return 0;
    s->nodes[s->sp++] = n;
    return 1;
}

__host__ __device__ __forceinline__ StackNode pop(MoveS *s) {
    return s->nodes[--s->sp];
}

__host__ __device__ void generate_multi(const Board *b, int is_white,
                                        MoveList *out) {
    if (!is_capture(out->moves[0]))
        return;

    MoveS st;
    st.sp = 0;

    // init stack with single-capture moves
    for (int i = 0; i < out->count; i++) {
        StackNode n;
        n.m = out->moves[i];
        memcpy(&n.board, b, sizeof(Board));
        apply_move(&n.board, n.m, is_white, 0);
        push(n, &st);
    }

    out->count = 0;
    // we don't need this to have size MOVELIST_SIZE, only 4
    SmallMoveList next;
    MoveListView v = view_from_small(&next);

    while (st.sp) {
        StackNode cur = pop(&st);

        next.count = 0;

        u32 last = move_end(b, cur.m, is_white);
        int found_capture = generate_single(&cur.board, is_white, &v, last);
        if (!found_capture) {
            append_move(out, cur.m);
            continue;
        }

        for (int i = 0; i < next.count; ++i) {
            Move extended = cur.m;
            Move step = next.moves[i];

            extended |= step;
            extended &= ~move_start(&cur.board, step, is_white);

            StackNode child;
            child.m = extended;
            memcpy(&child.board, &cur.board, sizeof(Board));
            apply_move(&child.board, step, is_white, 0);

            push(child, &st);
        }
    }
}

__host__ __device__ void generate_moves(const Board *b, int is_white,
                                        MoveList *out) {
    out->count = 0;

    MoveListView v = view_from_movelist(out);
    generate_single(b, is_white, &v, -1);
    force_captures(out);
    generate_multi(b, is_white, out);
    // force_captures(out);
}

void print_board(const Board *board, const MoveList *mlist, Move last_move,
                 int is_white) {
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
            u32 mask = 1u << idx;
            if (board->white & mask) {
                c = board->kings & mask ? 'W' : 'w';
            } else if (board->black & mask) {
                c = board->kings & mask ? 'B' : 'b';
            }
            int valid = 0;
            for (int i = 0; i < mlist->count; ++i) {
                if (move_start(board, mlist->moves[i], is_white) & mask) {
                    valid = 1;
                    break;
                }
            }
            if (valid)
                printf("%c ", c);
            else if (move_start(board, last_move, is_white) & mask ||
                     (move_end(board, last_move, is_white) & mask))
                printf(FORM_FADE FORM_UNDER "%c" FORM_END " ", c);
            else
                printf(FORM_FADE "%c " FORM_END, c);
            mask <<= 1;
        }
        printf("%d\n", 8 - y);
    }
    printf("  a b c d e f g h\n");
}

#endif /* MOVEGEN_H */
