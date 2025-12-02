#ifndef BOARD_H
#define BOARD_H

#include "stdint.h"
#include <stdio.h>
#ifdef INTRIN
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef FORMATTING
#define FORM_END "\x1b[m"
#define FORM_FADE "\x1b[2m"
#define FORM_UNDER "\x1b[4m"
#else
#define FORM_END ""
#define FORM_FADE ""
#define FORM_UNDER ""
#endif /* FORMATTING */

#define BOT_ROW 0x00041041
#define TOP_ROW 0x82000820

/*
Board representation:

  11  05  31  25
10  04  30  24
  03  29  23  17
02  28  22  16
  27  21  15  09
26  20  14  08
  19  13  07  01
18  12  06  00

Notice that every piece (except ones on edges) can move +1 or +7 up
and -1 or -7 down (mod 32)
Source: https://3dkingdoms.com/checkers/bitboards.htm#A1
*/
typedef uint32_t u32;
typedef uint8_t u8;

typedef struct {
    // positions of white pieces
    u32 white;
    // positions of black pieces
    u32 black;
    // positions of kings, either white or black
    u32 kings;
} Board;

#define CAN_UR 0x7DFDF5DD // can move +1
#define CAN_UL 0x79FBF3DB // can move +7
#define CAN_DR 0xFDF9EDBC // can move -7
#define CAN_DL 0xFBFBEBBA // can move -1

const int idx_to_board[32] = {
    11, 5,  31, 25, 10, 4,  30, 24, 3,  29, 23, 17, 2,  28, 22, 16,
    27, 21, 15, 9,  26, 20, 14, 8,  19, 13, 7,  1,  18, 12, 6,  0,
};

void init_board(Board *board) {
    board->kings = 0;
    board->white = 0xE3820C38; // top 3 rows
    board->black = 0x041C71C3; // bot 3 rows
}

__host__ __device__ __forceinline__ u32 rotl(u32 x, u8 n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_l(x, x, n);
#elif defined(INTRIN)
    return _rotl(x, n);
#else
    return (x << n) | (x >> ((32 - n) & 31));
#endif
}
__host__ __device__ __forceinline__ u32 rotr(u32 x, u8 n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_r(x, x, n);
#elif defined(INTRIN)
    return _rotr(x, n);
#else
    return (x >> n) | (x << ((32 - n) & 31));
#endif
}

#endif /* BOARD_H */
