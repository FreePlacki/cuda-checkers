#ifndef MOVE_H
#define MOVE_H

#include <stdint.h>

typedef uint32_t u32;
typedef uint8_t u8;

typedef struct {
    u32 captured; // mask for pieces captured by this move
    u8 path[10];  // sequence of visited squares: from, ..., to
    u8 path_len;  // number of squares in path (>=2)
} Move;

void simple_move(u8 from, u8 to, Move *out);

int is_capture(const Move *move);

// parse move string like "a3-b4" or "d2:f4:d6"
// returns 1 on success, 0 on syntax/validation error
int parse_move(const char *str, Move *out);

#define MOVE_STR_MAX 36
// given a move returns it's string representation
// out length has to be at least 12 * 3 = 36
void move_to_str(const Move *move, char *out);

#endif /* MOVE_H */
