#include "assert.h"
#include "board.cuh"
#include "gamestate.cuh"
#include "helpers.h"
#include "mcts.cuh"
#include "mcts_gpu.cuh"
#include "move.cuh"
#include "movegen.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void usage(char *pname) {
    fprintf(stderr,
            "Usage\t%s logfile.txt [init.txt]\ninit.txt - moves file to use "
            "for initial board state\n",
            pname);
    exit(1);
}

typedef enum {
    AI_RANDOM,
    AI_FLAT_MC_CPU,
    AI_FLAT_MC_GPU,
} AiLevel;

static AiLevel choose_ai_level() {
    for (;;) {
        printf("Choose AI level:\n");
        printf("1.\tRandom\n");
        printf("2.\tFlat Monte-Carlo (CPU)\n");
        printf("3.\tFlat Monte-Carlo (GPU)\n");

        char input[4];
        if (!fgets(input, sizeof(input), stdin))
            return AI_RANDOM;

        switch (input[0]) {
        case '1':
            return AI_RANDOM;
        case '2':
            return AI_FLAT_MC_CPU;
        case '3':
            return AI_FLAT_MC_GPU;
        }
        printf("Pick a number 1, 2 or 3\n");
    }
}

static Move choose_ai_move(GameState *game, const MoveList *mlist,
                           AiLevel lvl) {
    switch (lvl) {
    case AI_RANDOM:
        return choose_move_rand(game, mlist);
    case AI_FLAT_MC_CPU:
        return choose_move_flat_cpu(game, mlist);
    case AI_FLAT_MC_GPU:
        return choose_move_flat_gpu(game, mlist);
    default:
        return choose_move_rand(game, mlist);
    }
}

static int play_turn(GameState *game, FILE *logfile, int white_is_ai,
                     int black_is_ai, AiLevel white_ai_level,
                     AiLevel black_ai_level, int pause) {
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

    int is_white = (game->current_player == WHITE);
    int is_ai = is_white ? white_is_ai : black_is_ai;
    AiLevel lvl = is_white ? white_ai_level : black_ai_level;

    generate_moves(&game->board, is_white, &mlist);

    if (is_ai) {
        if (pause)
            print_board(&game->board, &mlist, &m);

        if (mlist.count == 0) {
            printf("NO VALID MOVES!\n");
            exit(1);
        }

        m = choose_ai_move(game, &mlist, lvl);

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

    apply_move(&game->board, &m, 1);
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

int player_v_player(GameState *g, FILE *log) {
    return play_turn(g, log, 0, 0, AI_RANDOM, AI_RANDOM, 0);
}

int player_black_v_ai(GameState *g, FILE *log) {
    static AiLevel white_ai;
    static int initialized = 0;
    if (!initialized) {
        white_ai = choose_ai_level();
        initialized = 1;
    }
    return play_turn(g, log, 1, 0, white_ai, AI_RANDOM, 0);
}

int player_white_v_ai(GameState *g, FILE *log) {
    static AiLevel black_ai;
    static int initialized = 0;
    if (!initialized) {
        black_ai = choose_ai_level();
        initialized = 1;
    }
    return play_turn(g, log, 0, 1, AI_RANDOM, black_ai, 0);
}

int ai_v_ai(GameState *g, FILE *log) {
    static AiLevel white_ai;
    static AiLevel black_ai;
    static int initialized = 0;

    if (!initialized) {
        printf("White AI:\n");
        white_ai = choose_ai_level();
        printf("Black AI:\n");
        black_ai = choose_ai_level();
        initialized = 1;
    }

    return play_turn(g, log, 1, 1, white_ai, black_ai, 1);
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
            return PLAYER_PLAYER;

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

    cudaError_t cudaStatus = cudaSetDevice(0);
    int noCudaDevice = cudaStatus != cudaSuccess;
    if (noCudaDevice) {
        printf("No CUDA device found!\n");
    }

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
