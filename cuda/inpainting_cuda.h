#ifndef __INPAINTING_CUDA_H__
#define __INPAINTING_CUDA_H__

#include <stdint.h>

class patch {
	public:
		int x; // width
		int y; // height
		int width;
		int height;
		patch() : x(0), y(0), width(0), height(0) {}
		patch(int xx, int yy, int ww, int hh) : x(xx), y(yy), width(ww), height(hh) {}
};

enum EPOS {
	UP_DOWN = 0,
	DOWN_UP,
	LEFT_RIGHT,
	RIGHT_LEFT,
	EPOS_COUNT,
};

void cuCalculateSSDTable(uint8_t *img, int w, int h, class patch *patch_list, int size, float *table);




#endif // __INPAINTING_CUDA_H__
