#include "movegen.h"
#include <string.h>

// int pos_to_index(char file, char rank);
// void index_to_xy(int idx, int *x, int *y);
//
// static inline int in_bounds(int x, int y) {
//     return x >= 0 && x < 8 && y >= 0 && y < 8 && ((x + y) & 1);
// }
//
// static inline int xy_to_index(int x, int y) { return y * 4 + (x / 2); }
//
// static const int dx[4] = {1, -1, 1, -1};
// static const int dy[4] = {1, 1, -1, -1};
//
// static inline void add_move(MoveList *list, int from, int to) {
//     if (list->count >= 128)
//         return;
//     Move *m = &list->moves[list->count++];
//     m->path_len = 2;
//     m->path[0] = (uint8_t)from;
//     m->path[1] = (uint8_t)to;
// }
//
// int generate_moves(const Board *b, int is_white, MoveList *out) {
//     out->count = 0;
//     int has_capture = 0;
//
//     u32 own = is_white ? b->white : b->black;
//     u32 opp = is_white ? b->black : b->white;
//     u32 kings = b->kings;
//     u32 occ = b->white | b->black;
//
//     for (int i = 0; i < 32; ++i) {
//         if (!(own & (1u << i)))
//             continue;
//
//         int is_king = (kings & (1u << i)) != 0;
//         int x, y;
//         index_to_xy(i, &x, &y);
//
//         for (int d = 0; d < 4; ++d) {
//             // Men move forward only
//             if (!is_king) {
//                 if (is_white && dy[d] < 0)
//                     continue;
//                 if (!is_white && dy[d] > 0)
//                     continue;
//             }
//
//             int nx = x + dx[d];
//             int ny = y + dy[d];
//             if (!in_bounds(nx, ny))
//                 continue;
//             int nidx = xy_to_index(nx, ny);
//             u32 nmask = 1u << nidx;
//
//             if (!(occ & nmask)) {
//                 // Empty square — simple move
//                 if (!has_capture)
//                     add_move(out, i, nidx);
//             } else if (opp & nmask) {
//                 // Check capture jump
//                 int jx = nx + dx[d];
//                 int jy = ny + dy[d];
//                 if (!in_bounds(jx, jy))
//                     continue;
//                 int jidx = xy_to_index(jx, jy);
//                 u32 jmask = 1u << jidx;
//                 if (!(occ & jmask)) {
//                     // Capture found
//                     if (!has_capture) {
//                         out->count = 0;
//                         has_capture = 1;
//                     }
//                     add_move(out, i, jidx);
//                 }
//             }
//         }
//     }
//
//     return out->count;
// }
