#include "board.h"
#include "gamestate.h"
#include "helpers.h"
#include "move.h"
#include "movegen.h"
#include "assert.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void usage(char *pname) {
    fprintf(stderr,
            "Usage\t%s logfile.txt [init.txt]\ninit.txt - moves file to use "
            "for initial board state",
            pname);
    exit(1);
}

Move choose_move(const Board *board, const MoveList *l) {
    int r = rand() % l->count;
    return l->moves[r];
}

static int play_turn(GameState *game, FILE *logfile, int is_ai, int pause) {
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

    char input[MOVE_STR_MAX];
    static Move m;
    MoveList mlist;
    int is_white = game->current_player == WHITE;
    generate_moves(&game->board, is_white, &mlist);

    if (is_ai) {
        if (pause)
            print_board(&game->board, &mlist, &m);
        m = choose_move(&game->board, &mlist);
        move_to_str(&m, input);
        printf("AI (%c) chose %s\n", is_white ? 'w' : 'b', input);
        assert(is_valid_move(&m, &mlist));
    } else {
        print_board(&game->board, &mlist, &m);
        printf("\nPossible moves:\n");
        print_movelist(&mlist);

        printf("\nPlayer (%c) move: ", is_white ? 'w' : 'b');
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
    }

    apply_move(&game->board, &m);
    move_to_str(&m, input);
    if (logfile)
        fprintf(logfile, "%s\n", input);

    next_turn(game, is_capture(&m));

    if (pause) {
        printf("Press ENTER to continue...\n");
        fgets(input, sizeof(input), stdin);
    }
    return 0;
}

int player_v_player(GameState *game, FILE *logfile) {
    return play_turn(game, logfile, 0, 0);
}

int player_black_v_ai(GameState *game, FILE *logfile) {
    int is_ai = (game->current_player == WHITE);
    return play_turn(game, logfile, is_ai, 0);
}

int player_white_v_ai(GameState *game, FILE *logfile) {
    int is_ai = (game->current_player == BLACK);
    return play_turn(game, logfile, is_ai, 0);
}

int ai_v_ai(GameState *game, FILE *logfile) {
    return play_turn(game, logfile, 1, 1);
}

typedef enum {
    PLAYER_PLAYER,
    PLAYER_BLACK_AI,
    PLAYER_WHITE_AI,
    AI_AI,
} GameMode;

GameMode choose_game_mode() {
    for (;;) {
        printf("Choose mode:\n1.\tPlayer vs Player\n2.\tPlayer (black) vs "
               "AI\n3.\tPlayer (white) vs AI\n4.\tAI vs AI\n");
        char input[4];
        if (!fgets(input, sizeof(input), stdin))
            return 1;

        switch (input[0]) {
        case '1':
            return PLAYER_PLAYER;
        case '2':
            return PLAYER_BLACK_AI;
        case '3':
            return PLAYER_WHITE_AI;
        case '4':
            return AI_AI;
        }
        printf("Pick a number 1, 2, 3 or 4\n");
    }
}

int main(int argc, char **argv) {
    srand(time(0));

    GameState game;
    init_game(&game);

    char *pname = argv[0];
    if (!(argc >= 2 && argc <= 3))
        usage(pname);

    FILE *logfile = fopen(argv[1], "w");
    if (!logfile) {
        perror("Couldn't open log file");
        return 1;
    }

    if (argc == 3) {
        FILE *init_file = fopen(argv[2], "r");
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

    int (*play)(GameState *, FILE *);
    switch (choose_game_mode()) {
    case PLAYER_PLAYER:
        play = player_v_player;
        break;
    case PLAYER_BLACK_AI:
        play = player_black_v_ai;
        break;
    case PLAYER_WHITE_AI:
        play = player_white_v_ai;
        break;
    case AI_AI:
        play = ai_v_ai;
        break;
    }

    for (;;) {
        if (play(&game, logfile))
            break;
    }

    fclose(logfile);
    return 0;
}
