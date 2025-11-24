#ifndef TREE_H
#define TREE_H

#include "float.h"
#include "math.h"

typedef struct Node Node;

typedef struct {
    u32 count;
    u32 size;

    Node **nodes;
} Nodes;

typedef struct Node {
    // # games played * 2
    u32 games_played;
    // # games won * 2 + # draws * 1
    u32 games_won;

    struct Node *parent;
    Nodes children;
    GameState gs;
} Node;


#define INITIAL_NODES_SZ 4
int nodes_init(Nodes *nodes) {
    nodes->count = 0;
    nodes->size = INITIAL_NODES_SZ;

    nodes->nodes = (Node**)malloc(INITIAL_NODES_SZ * sizeof(Node *));
    if (!nodes->nodes)
        return 0;

    return 1;
}

int nodes_add(Nodes *nodes, Node *node) {
    if (nodes->size == nodes->count) {
        nodes->size <<= 1;
        nodes->nodes = (Node**)realloc(nodes->nodes, nodes->size * sizeof(Node *));
        if (!nodes->nodes)
            return 0;
    }
    nodes->nodes[nodes->count++] = node;
    return 1;
}

void nodes_free(Nodes *nodes) {
    if (nodes && nodes->nodes)
        free(nodes->nodes);
    nodes->size = nodes->count = 0;
}

Node *node_init(Node *parent, GameState gs) {
    Node *n = (Node *)malloc(sizeof(Node));

    n->games_played = n->games_won = 0;
    n->parent = parent;
    if (!nodes_init(&n->children)) {
        free(n);
        return NULL;
    }
    n->gs = gs;

    return n;
}

void node_free(Node *n) {
    if (!n)
        return;
    for (int i = 0; i < n->children.count; ++i)
        node_free(n->children.nodes[i]);
    nodes_free(&n->children);
    free(n);
}

int node_is_terminal(const Node *n) { return game_result(&n->gs) != PENDING; }

void node_backprop(Node *n, GameResult result) {
    // IMPORTANT: since node stores the GameState *after* the move was made
    // we have to flip the current_player here to make the node represent the
    // correct side
    int is_white = n->gs.current_player != WHITE;
    switch (result) {
    case WHITE_WON:
        if (is_white)
            n->games_won += 2;
        break;
    case BLACK_WON:
        if (!is_white)
            n->games_won += 2;
        break;
    case DRAW:
        n->games_won += 1;
        break;
    case PENDING:
        assert(0 && "unreachable");
    }
    n->games_played += 2;

    if (n->parent)
        node_backprop(n->parent, result);
}

#define VALUATION_C 1.414213562373095f // sqrt(2)
float node_valuation(const Node *n) {
    assert(n->parent);

    if (n->games_played == 0)
        return FLT_MAX;

    float exploitation = (float)n->games_won / (float)n->games_played;
    float exploration =
        VALUATION_C *
        sqrtf(log((float)n->parent->games_played) / (float)n->games_played);

    return exploitation + exploration;
}

Node *node_select_child(Node *root) {
    float best_v = 0.0f;
    Node *best_n = NULL;
    for (int i = 0; i < root->children.count; ++i) {
        Node *ch = root->children.nodes[i];
        float val = node_valuation(ch);
        if (val > best_v) {
            best_v = val;
            best_n = ch;
        }
    }

    return best_n;
}

Node *node_select_leaf(Node *root) {
    assert(!root->parent);

    Node *n = root;
    for (;;) {
        if (node_is_terminal(n))
            return n;

        if (n->children.count == 0)
            return n;

        Node *ch = node_select_child(n);
        if (!ch)
            return n;
        n = ch;
    }
}

Node *node_expand(Node *n, const MoveList *l) {
    for (int i = 0; i < l->count; ++i) {
        GameState gs = n->gs;
        apply_move(&gs.board, l->moves[i], gs.current_player == WHITE, 1);
        next_turn(&gs, is_capture(l->moves[i]));

        Node *child = node_init(n, gs);
        if (!child)
            return NULL;
        nodes_add(&n->children, child);
    }

    return n->children.nodes[0];
}

#endif /* TREE_H */
