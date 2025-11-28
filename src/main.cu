#include "assert.h"
#include "board.cuh"
#include "gamestate.cuh"
#include "helpers.h"
#include "mcts.h"
#include "mcts_gpu.cuh"
#include "move.cuh"
#include "movegen.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

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
    AI_MCTS_CPU,
    AI_MCTS_GPU,
} AiLevel;

typedef struct {
    AiLevel level;
    double timeout;
} AiPlayer;

static double choose_ai_timeout() {
    char line[64];
    double timeout = 1.0; // default value

    for (;;) {
        printf("Choose AI time limit [s] (default %.1lf s): ", timeout);
        if (!fgets(line, sizeof(line), stdin)) {
            return timeout;
        }

        line[strcspn(line, "\n")] = 0;

        if (strlen(line) == 0) {
            return timeout;
        }

        char *endptr;
        double val = strtod(line, &endptr);

        if (endptr == line || *endptr != '\0') {
            printf("Invalid input, please enter a number.\n");
            continue;
        }

        if (val <= 0.0) {
            printf("Timeout must be positive.\n");
            continue;
        }

        return val;
    }
}

static AiPlayer choose_ai_level() {
    double timeout = -1.0;
    for (;;) {
        printf("Choose AI level:\n");
        printf("1.\tRandom\n");
        printf("2.\tFlat Monte-Carlo (CPU)\n");
        printf("3.\tFlat Monte-Carlo (GPU)\n");
        printf("4.\tMonte-Carlo Tree Search (CPU)\n");
        printf("5.\tMonte-Carlo Tree Search (GPU)\n");

        AiPlayer pl;
        pl.level = AI_RANDOM;
        pl.timeout = timeout;

        char input[4];
        if (!fgets(input, sizeof(input), stdin))
            return pl;

        switch (input[0]) {
        case '1':
            return pl;
        case '2':
            pl.level = AI_FLAT_MC_CPU;
            return pl;
        case '3':
            pl.level = AI_FLAT_MC_GPU;
            return pl;
        case '4':
            pl.level = AI_MCTS_CPU;
            pl.timeout = choose_ai_timeout();
            return pl;
        case '5':
            pl.level = AI_MCTS_GPU;
            pl.timeout = choose_ai_timeout();
            return pl;
        }
        printf("Pick a number 1, 2, 3, 4 or 5\n");
    }
}

static Move choose_ai_move(GameState *game, const MoveList *mlist,
                           AiPlayer player) {
    switch (player.level) {
    case AI_RANDOM:
        return choose_move_rand(game, mlist);
    case AI_FLAT_MC_CPU:
        return choose_move_flat_cpu(game, mlist);
    case AI_FLAT_MC_GPU:
        return choose_move_flat_gpu(game, mlist);
    case AI_MCTS_CPU:
        return choose_move_cpu(*game, mlist, player.timeout);
    case AI_MCTS_GPU:
        return choose_move_gpu(*game, mlist, player.timeout);
    default:
        assert(0 && "unreachable");
        return choose_move_rand(game, mlist);
    }
}

static int play_turn(GameState *game, FILE *logfile, int white_is_ai,
                     int black_is_ai, AiPlayer white_ai, AiPlayer black_ai,
                     int pause) {
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
    int is_ai = is_white ? white_is_ai : black_is_ai;
    AiPlayer lvl = is_white ? white_ai : black_ai;

    generate_moves(&game->board, is_white, &mlist);

    if (is_ai) {
        print_board(&game->board, &mlist, m, is_white);

        if (pause) {
            printf("Press ENTER to continue...\n");
            fgets(input, sizeof(input), stdin);
        }

        if (mlist.count == 0) {
            if (is_white)
                printf("Black won!\n");
            else
                printf("White won!\n");
            return 1;
        }

        m = choose_ai_move(game, &mlist, lvl);

        move_to_str(&game->board, m, is_white, input);
        printf("AI (%c) chose %s\n", is_white ? 'w' : 'b', input);
        assert(is_valid_move(m, &mlist));

    } else {
        print_board(&game->board, &mlist, m, is_white);
        printf("\nPossible moves:\n");
        print_movelist(&game->board, &mlist, is_white);

        printf("\nPlayer (%c) move: ", is_white ? 'w' : 'b');
        if (!fgets(input, sizeof(input), stdin))
            return 1;
        input[strcspn(input, "\n")] = 0;

        if (!parse_move(input, &m)) {
            printf("Incorrect move format.\n");
            return 0;
        }
        if (!is_valid_move(m, &mlist)) {
            printf("Invalid move.\n");
            return 0;
        }
    }

    move_to_str(&game->board, m, is_white, input);
    apply_move(&game->board, m, is_white, 1);
    if (logfile) {
        fprintf(logfile, "%s\n", input);
        fflush(logfile);
    }

    next_turn(game, is_capture(m));

    return 0;
}

int player_v_player(GameState *g, FILE *log) {
    AiPlayer dummy;
    dummy.timeout = -1.0;
    dummy.level = AI_RANDOM;
    return play_turn(g, log, 0, 0, dummy, dummy, 0);
}

int player_black_v_ai(GameState *g, FILE *log) {
    static AiPlayer white_ai;
    static int initialized = 0;
    if (!initialized) {
        white_ai = choose_ai_level();
        initialized = 1;
    }
    return play_turn(g, log, 1, 0, white_ai, white_ai, 0);
}

int player_white_v_ai(GameState *g, FILE *log) {
    static AiPlayer black_ai;
    static int initialized = 0;
    if (!initialized) {
        black_ai = choose_ai_level();
        initialized = 1;
    }
    return play_turn(g, log, 0, 1, black_ai, black_ai, 0);
}

int ai_v_ai(GameState *g, FILE *log) {
    static AiPlayer white_ai;
    static AiPlayer black_ai;
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
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

    cudaError_t cudaStatus = cudaSetDevice(0);
    int noCudaDevice = cudaStatus != cudaSuccess;
    if (noCudaDevice) {
        printf("cudaSetDevice failed: %d (%s)\n", cudaStatus,
               cudaGetErrorString(cudaStatus));
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
