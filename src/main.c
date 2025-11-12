#include "board.h"
#include "gamestate.h"
#include "helpers.h"
#include "move.h"
#include "movegen.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void usage(char *pname) {
    fprintf(stderr,
            "Usage\t%s [init.txt]\ninit.txt - moves file to use for initial "
            "board state",
            pname);
    exit(1);
}

int player_v_player(GameState *game, FILE *logfile) {
    MoveList mlist;
    Move m;
    switch (game_result(game)) {
    case WHITE_WON:
        printf("White won!\n");
        return 1;
    case BLACK_WON:
        printf("Black won!\n");
        return 1;
    case DRAW:
        printf("Draw!\n");
        return 1;
    case PENDING:
        break;
    }
    print_board(&game->board);

    printf("\nPossible moves:\n");
    generate_moves(&game->board, game->current_player == WHITE, &mlist);
    print_movelist(&mlist);

    printf("\nPlayer %d (%c) move: ", game->current_player,
           game->current_player == WHITE ? 'w' : 'b');
    char input[MOVE_STR_MAX];
    if (!fgets(input, sizeof(input), stdin))
        return 1;
    input[strcspn(input, "\n")] = 0;

    if (!parse_move(input, &m)) {
        printf("Incorrect move format.\n");
        return 0;
    }

    if (!is_valid_move(&m, &mlist)) {
        printf("Invalid move.\n");
        return 0;
    }

    apply_move(&game->board, &m);

    move_to_str(&m, input);
    if (logfile)
        fprintf(logfile, "%s\n", input);
    game->current_player = !game->current_player;

    return 0;
}

int main(int argc, char **argv) {
    GameState game;
    init_game(&game);

    char *pname = argv[0];
    if (argc > 2)
        usage(pname);

    FILE *logfile = fopen("game_log.txt", "w");
    if (!logfile) {
        perror("Couldn't open log file");
        return 1;
    }

    if (argc == 2) {
        FILE *init_file = fopen(argv[1], "r");
        if (!init_file) {
            perror("Couldn't open init file");
            return 1;
        }
        if (!seed_game(&game, init_file, logfile)) {
            perror("Invalid move in init file");
            return 1;
        }
        printf("Initialized game state from input file\n");
    }

    for (;;) {
        if (player_v_player(&game, logfile))
            break;
    }

    fclose(logfile);
    return 0;
}
