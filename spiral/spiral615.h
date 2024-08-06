#pragma once

struct spiral615;

spiral615 *create_spiral615(const int *poly, int len);
int init_spiral615(spiral615 *vp, int starting_state);
void update_spiral615(spiral615 *vp, unsigned char *syms, int nbits);
int chainback_spiral615(spiral615 *vp, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_spiral615(spiral615 *vp);
