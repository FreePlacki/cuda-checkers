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

void init_game(GameState *state) {
    state->current_player = BLACK;
    init_board(&state->board);
}

void log_move(const char *notation, FILE *logfile) {
    if (logfile)
        fprintf(logfile, "%s\n", notation);
}

int seed_game(GameState *gs, FILE *f) {
    char buf[MOVE_STR_MAX];

    Move m;
    while (fgets(buf, MOVE_STR_MAX, f)) {
        if (!parse_move(buf, &m))
            return 0;
        apply_move(&gs->board, &m);
        gs->current_player = !gs->current_player;
    }
    return 1;
}

void usage(char *pname) {
    fprintf(stderr,
            "Usage\t%s [init.txt]\ninit.txt - moves file to use for initial "
            "board state",
            pname);
    exit(1);
}

int main(int argc, char **argv) {
    GameState game;
    init_game(&game);

    char *pname = argv[0];
    if (argc > 2)
        usage(pname);
    if (argc == 2) {
        FILE *init_file = fopen(argv[1], "r");
        if (!init_file) {
            perror("Couldn't open init file");
            return 1;
        }
        if (!seed_game(&game, init_file)) {
            perror("Invalid move in init file");
            return 1;
        }
        printf("Initialized game state from input file\n");
    }

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

        printf("\nPlayer %d (%c) move: ", game.current_player,
               game.current_player == WHITE ? 'w' : 'b');
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

        move_to_str(&move, input);
        log_move(input, logfile);
        game.current_player = !game.current_player;
    }

    fclose(logfile);
    return 0;
}
