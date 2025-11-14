#include "movegen.h"
#include "assert.h"
#include "board.h"
#include "helpers.h"
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
    if (l->count == 0)
        return;
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

#define CAN_UR 0x7DFDF5DD // can move +1
#define CAN_UL 0x79FBF3DB // can move +7
#define CAN_DR 0xFDF9EDBC // can move -7
#define CAN_DL 0xFBFBEBBA // can move -1

static inline u32 rotl(u32 x, u8 n) { return (x << n) | (x >> (32 - n)); }
static inline u32 rotr(u32 x, u8 n) { return (x >> n) | (x << (32 - n)); }

static void gen_shift_moves(u32 own, u32 free, int shift, MoveList *out) {
    u32 shifted = shift > 0 ? rotl(own, shift) : rotr(own, -shift);

    u32 moves = shifted & free;

    // iterating 1s (from least significant)
    while (moves) {
        u32 dst = moves & -moves;
        moves ^= dst;

        // number of consecutive 0s (form least significant)
        int dst_idx = __builtin_ctz(dst);
        int src_idx = (dst_idx + 32 - shift) % 32;

        Move m;
        simple_move(src_idx, dst_idx, &m);
        append_move(out, m);
    }
}

static int gen_shift_capture(u32 ownp, u32 ene, u32 free, int step,
                             MoveList *out) {
    u32 nei = step > 0 ? rotl(ownp, step) : rotr(ownp, -step);
    nei &= ene;
    u32 dst = (step > 0) ? rotl(nei, step) : rotr(nei, -step);
    dst &= free;

    int cap_found = dst != 0;
    while (dst) {
        u32 dst_b = dst & -dst;
        dst ^= dst_b;

        int dst_idx = __builtin_ctz(dst_b);
        int cap_idx = (dst_idx + 32 - step) % 32;
        int src_idx = (cap_idx + 32 - step) % 32;

        Move m;
        simple_move(src_idx, dst_idx, &m);
        m.captured = 1u << cap_idx;
        append_move(out, m);
    }

    return cap_found;
}

static void generate_single(const Board *b, int is_white, MoveList *out) {
    out->count = 0;

    u32 own = is_white ? b->white : b->black;
    u32 ene = is_white ? b->black : b->white;
    u32 kings = b->kings & own;
    u32 men = own & ~kings;
    u32 occ = own | ene;
    u32 free = ~occ;

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
}

static void generate_multi(const Board *b, int is_white, MoveList *out) {
    assert(out->count > 0);
    force_captures(out);
    if (!is_capture(&out->moves[0]))
        return;

    Board tboard;
    memcpy(&tboard, b, sizeof(Board));
    for (int i = 0; i < out->count; ++i) {
        apply_move(&tboard, &out->moves[i]);
        // TODO
    }
}

void generate_moves(const Board *b, int is_white, MoveList *out) {
    out->count = 0;

    generate_single(b, is_white, out);
    // force_captures(out);
}
