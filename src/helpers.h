#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>

#define dbg(x) ({ __typeof__(x) _v = (x); fprintf(stderr, "%s:%d: %s = %lld\n", \
    __FILE__, __LINE__, #x, (long long)_v); _v; })

#endif /* HELPERS_H */
