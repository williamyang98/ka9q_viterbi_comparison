#pragma once

struct spiral29;

spiral29 *create_spiral29(const int *poly, int len);
int init_spiral29(spiral29 *vp, int starting_state);
void update_spiral29(spiral29 *vp, unsigned char *syms, int nbits);
int chainback_spiral29(spiral29 *vp, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_spiral29(spiral29 *vp);
