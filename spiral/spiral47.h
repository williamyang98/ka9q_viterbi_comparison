#pragma once

struct spiral47;

spiral47 *create_spiral47(const int *poly, int len);
int init_spiral47(spiral47 *vp, int starting_state);
void update_spiral47(spiral47 *vp, unsigned char *syms, int nbits);
int chainback_spiral47(spiral47 *vp, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_spiral47(spiral47 *vp);
