#include "board.h"
#include "helpers.h"
#include "move.h"
#include "movegen.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    BLACK,
    WHITE,
} Player;

typedef struct {
    Board board;
    unsigned char current_player;
} GameState;

void init_game(GameState *state);
void log_move(const char *notation, FILE *logfile);

void init_game(GameState *state) {
    state->current_player = BLACK;
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

    MoveList mlist;
    Move move;
    for (;;) {
        print_board(&game.board);

        printf("\nPossible moves:\n");
        generate_moves(&game.board, game.current_player == WHITE, &mlist);
        print_movelist(&mlist);

        printf("\nPlayer %d move: ", game.current_player);
        char input[MOVE_STR_MAX];
        if (!fgets(input, sizeof(input), stdin))
            break;
        input[strcspn(input, "\n")] = 0;

        if (!parse_move(input, &move)) {
            printf("Incorrect move format.\n");
            continue;
        }

        if (!is_valid_move(&move, &mlist)) {
            printf("Invalid move.\n");
            continue;
        }

        apply_move(&game.board, &move);

        log_move(input, logfile);
        game.current_player = !game.current_player;
    }

    fclose(logfile);
    return 0;
}
