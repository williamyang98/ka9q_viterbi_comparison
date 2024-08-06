#pragma once

struct spiral49;

spiral49 *create_spiral49(const int *poly, int len);
int init_spiral49(spiral49 *vp, int starting_state);
void update_spiral49(spiral49 *vp, unsigned char *syms, int nbits);
int chainback_spiral49(spiral49 *vp, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_spiral49(spiral49 *vp);
