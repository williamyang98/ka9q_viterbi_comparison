#pragma once

struct spiral27;

spiral27 *create_spiral27(const int *poly, int len);
int init_spiral27(spiral27 *vp, int starting_state);
void update_spiral27(spiral27 *vp, unsigned char *syms, int nbits);
int chainback_spiral27(spiral27 *vp, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_spiral27(spiral27 *vp);
