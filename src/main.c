#include "board.h"
#include "move.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    WHITE,
    BLACK,
} Player;

typedef struct {
    Board board;
    unsigned char current_player;
} GameState;

void init_game(GameState *state);
int generate_moves(const GameState *state, int moves[][4], int max_moves);
void log_move(const char *notation, FILE *logfile);

void evaluate_position_gpu(const GameState *state) {
    // TODO
}

void init_game(GameState *state) {
    state->current_player = WHITE;
    init_board(&state->board);
}

void log_move(const char *notation, FILE *logfile) {
    if (logfile)
        fprintf(logfile, "%s\n", notation);
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
        print_board(&game.board);
        printf("Player %d move: ", game.current_player);
        char input[32];
        if (!fgets(input, sizeof(input), stdin))
            break;
        input[strcspn(input, "\n")] = 0;

        Move move;
        if (!parse_move(input, &move)) {
            printf("Incorrect move.\n");
            continue;
        }

        apply_move(&game.board, &move);

        log_move(input, logfile);
        evaluate_position_gpu(&game);
        game.current_player = !game.current_player;
    }

    fclose(logfile);
    return 0;
}
