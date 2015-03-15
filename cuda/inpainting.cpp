#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_inpainting.h"


using namespace std;
using namespace cv;

#define RADIUS	(16)
#define RANGE_RATIO	(2.0f)

const int PATCH_WIDTH = RADIUS;
const int PATCH_HEIGHT = RADIUS;
const int NODE_WIDTH = PATCH_WIDTH / 2;
const int NODE_HEIGHT = PATCH_HEIGHT / 2;

const float CONST_FULL_MSG = PATCH_HEIGHT * PATCH_WIDTH * 255 * 255 * 3 / 2;
float FULL_MSG = 0;


int main(int argc, char **argv) {
	if(argc != 8) {
		cout<<"Usage: "<<argv[0]<<" input x y w h output iter_time"<<endl;
		return 0;
	}
	// construct a CudaInpainting class preparing for the coming inpainting parameters
	CudaInpainting ci(argv[1]);

	// parse the arguments
	char *input = argv[1],
	     *output = argv[6];
	int maskX = atoi(argv[2]),
	    maskY = atoi(argv[3]),
	    maskW = atoi(argv[4]),
	    maskH = atoi(argv[5]),
	    iterTime = atoi(argv[7]);
	cout << "Begin to Inpainting" << endl;

	// invoke the inpainting function 
	ci.Inpainting(maskX, maskY, maskW, maskH, iterTime);
	cout << "Begin to write the image" << endl;

	// write the output image to the output file
	imwrite(output, ci.GetImage());
	cout << "Done" << endl;
	
	return 0;
}
