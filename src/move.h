#ifndef MOVE_H
#define MOVE_H

#include <stdint.h>

typedef uint32_t u32;
typedef uint8_t u8;


typedef struct {
    u8 path_len;     // number of squares in path (>=2)
    u8 path[12];     // sequence of visited squares: from, ..., to
} Move;

// parse move string like "a3-b4" or "d2:f4:d6"
// returns 1 on success, 0 on syntax/validation error
int parse_move(const char *str, Move *out);

#endif /* MOVE_H */
