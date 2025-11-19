#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include "stdint.h"

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

const int idx_to_board[32] = {
    11, 5,  31, 25, 10, 4,  30, 24, 3,  29, 23, 17, 2,  28, 22, 16,
    27, 21, 15, 9,  26, 20, 14, 8,  19, 13, 7,  1,  18, 12, 6,  0,
};

void init_board(Board *board) {
    board->kings = 0;
    board->white = 0xE3820C38; // top 3 rows
    board->black = 0x041C71C3; // bot 3 rows
}

#endif /* BOARD_H */
