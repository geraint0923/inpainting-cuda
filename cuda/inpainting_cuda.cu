
#include "inpainting_cuda.h"


void cuCalculateSSDTable(uint8_t *img, int w, int h, class patch *patch_list, int size, float *table);
