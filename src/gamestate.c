#include "gamestate.h"
#include "move.h"
void init_game(GameState *state) {
    state->current_player = BLACK;
    init_board(&state->board);
}

int seed_game(GameState *gs, FILE *f, FILE *logfile) {
    char buf[MOVE_STR_MAX];

    Move m;
    while (fgets(buf, MOVE_STR_MAX, f)) {
        if (!parse_move(buf, &m))
            return 0;
        apply_move(&gs->board, &m);
        gs->current_player = !gs->current_player;

        move_to_str(&m, buf);
        if (logfile)
            fprintf(logfile, "%s\n", buf);
    }
    return 1;
}

#define MAX_NOCAPTURE 50
GameResult game_result(const GameState *gs) {
    if (gs->no_capture >= MAX_NOCAPTURE)
        return DRAW;
    if (gs->board.white == 0u)
        return BLACK_WON;
    if (gs->board.white == 0u)
        return WHITE_WON;
    return PENDING;
}
