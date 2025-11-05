#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BOARD_SIZE 8

typedef enum {
    BLACK = 1,
    WHITE = 2,
} Player;

typedef enum {
    EMPTY = ' ',
    BLACK_PAWN = 'b',
    BLACK_KING = 'B',
    WHITE_PAWN = 'w',
    WHITE_KING = 'W',
} Tile;

typedef struct {
    unsigned char board[BOARD_SIZE * BOARD_SIZE / 2];
    unsigned char current_player;
} GameState;

void init_game(GameState *state);
void print_board(const GameState *state);
int parse_move(const char *input, int *from_x, int *from_y, int *to_x, int *to_y);
int make_move(GameState *state, int from_x, int from_y, int to_x, int to_y);
int generate_moves(const GameState *state, int moves[][4], int max_moves);
void log_move(const char *notation, FILE *logfile);
static inline int pos_index(int x, int y);

// convert (x, y) on 8x8 board to 0–31 index
// valid only for playable (dark) squares
static inline int pos_index(int x, int y) {
    if ((x + y) % 2 == 0) {
        return -1; // non-playable tile
    }
    return (y * 4) + (x / 2);
}

void evaluate_position_gpu(const GameState *state) {
    // todo
}

void init_game(GameState *state) {
    printf("%ld\n", sizeof(state->board));
    memset(state->board, EMPTY, sizeof(state->board));
    state->current_player = BLACK;

    for (int y = 0; y < 3; ++y)
        for (int x = (y + 1) % 2; x < BOARD_SIZE; x += 2)
            state->board[pos_index(x, y)] = BLACK_PAWN;

    for (int y = 5; y < 8; ++y)
        for (int x = (y + 1) % 2; x < BOARD_SIZE; x += 2)
            state->board[pos_index(x, y)] = WHITE_PAWN;
}

void print_board(const GameState *state) {
    printf("\n  a b c d e f g h\n");
    for (int y = 0; y < BOARD_SIZE; ++y) {
        printf("%d ", 8 - y);
        for (int x = 0; x < BOARD_SIZE; ++x) {
            int pos = pos_index(x, y);
            int p = EMPTY;
            if (pos != -1) {
                p = state->board[pos];
            }
            char c = '.';
            if (p == WHITE_PAWN) c = 'w';
            else if (p == BLACK_PAWN) c = 'b';
            else if (p == WHITE_KING) c = 'W';
            else if (p == BLACK_KING) c = 'B';
            printf("%c ", c);
        }
        printf("%d\n", 8 - y);
    }
    printf("  a b c d e f g h\n");
}

int parse_move(const char *input, int *from_x, int *from_y, int *to_x, int *to_y) {
    if (strlen(input) < 5) return 0;
    *from_x = input[0] - 'a';
    *from_y = 8 - (input[1] - '0');
    *to_x   = input[3] - 'a';
    *to_y   = 8 - (input[4] - '0');
    return 1;
}

int make_move(GameState *state, int from_x, int from_y, int to_x, int to_y) {
    int piece = state->board[pos_index(from_x, from_y)];
    if (!piece) return 0;
    state->board[pos_index(to_x, to_y)] = piece;
    state->board[pos_index(from_x, from_y)] = EMPTY;
    return 1;
}

int generate_moves(const GameState *state, int moves[][4], int max_moves) {
    if (max_moves < 1) return 0;
    moves[0][0] = 1; moves[0][1] = 5; moves[0][2] = 2; moves[0][3] = 4;
    return 1;
}

void log_move(const char *notation, FILE *logfile) {
    if (logfile) fprintf(logfile, "%s\n", notation);
}

int main(int argc, char **argv) {
    GameState game;
    init_game(&game);

    FILE *logfile = fopen("game_log.txt", "w");
    if (!logfile) {
        perror("Couldn't open log file");
        return 1;
    }

    for (;;) {
        print_board(&game);
        printf("Player %d move: ", game.current_player);
        char input[32];
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;

        int fx, fy, tx, ty;
        if (!parse_move(input, &fx, &fy, &tx, &ty)) {
            printf("Incorrect format.\n");
            continue;
        }

        if (!make_move(&game, fx, fy, tx, ty)) {
            printf("Incorrect format.\n");
            continue;
        }

        log_move(input, logfile);
        evaluate_position_gpu(&game);
        game.current_player = 3 - game.current_player;
    }

    fclose(logfile);
    return 0;
}

