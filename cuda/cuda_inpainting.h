#ifndef __INPAINTING_CUDA_H__
#define __INPAINTING_CUDA_H__

#include <stdint.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



// the class for the inpainting operation
class CudaInpainting {
public:
	// constructor using the input image file
	CudaInpainting(const char *path);

	// the function to do the inpainting job
	bool Inpainting(int x, int y, int width, int height, int iterTime);

	// get the output image
	cv::Mat GetImage();

	// destructor: clean up the allocated memory on CPU and GPU
	~CudaInpainting();

	// the enumerate variables to indicate the relative position of two patches
	enum EPOS {
		UP_DOWN = 0,
		DOWN_UP,
		LEFT_RIGHT,
		RIGHT_LEFT,
		EPOS_COUNT,
	};
	
	// the enumerate variables to indicate the message passing directions
	enum EDIR {
		DIR_UP = 0, 
		DIR_DOWN,
		DIR_LEFT,
		DIR_RIGHT,
		DIR_COUNT,
	};

	// the parameters to be used when doing the inpainting job
	static const int RADIUS;
	static const float RANGE_RATIO;

	static const int PATCH_WIDTH;
	static const int PATCH_HEIGHT;
	static const int NODE_WIDTH;
	static const int NODE_HEIGHT;
	static const float CONST_FULL_MSG;


	// this class represent the patch to be extract from the input image
	class Patch {
	public:
		int x;		// the x coordinate of the left top of this patch
		int y;		// the y coordiante of the right bottom of this patch
		int width;	// the width of this patch
		int height;	// the height of this patch
		Patch() : x(0), y(0), width(0), height(0) {}
		Patch(int ww, int hh);
		Patch(int xx, int yy, int ww, int hh) : x(xx), y(yy), width(ww), height(hh) {}
	};

	// the node class used in the nodetable
	class Node {
	public:
		int x;
		int y;
	};

	// the SSD entry including four directions: up, down, left, right
	class SSDEntry {
	public:
		float data[EPOS_COUNT];
	};

private:
	// member function
	
	// round up the mask region to make edge larger
	Patch RoundUpArea(Patch p);

	// to judge if two patches have overlap region
	bool OverlapPatch(Patch &p1, Patch &p2);

	// generate the candidate patches list
	void GenPatches();

	// calculate the SSD for specified pair of patches
	// this function is wrap of GPU function
	float CalculateSSD(Patch& p1, Patch& p2, EPOS pos);

	// calculate the SSD table for all pairs of patches
	// this function is also a wrap of GPU global function
	void CalculateSSDTable();

	// initialize the node table
	// including the message to be passed in the following iteration phase
	void InitNodeTable();

	// the iteration function
	void RunIteration(int times);

	// the calculate the belief in each node for each patch
	// to find out the best match patch for the current node
	void SelectPatch();

	// fille the best match patch into the region of the corresponding node
	void FillPatch();

	// a helper to make the process of filling the target region easier
	static void PastePatch(cv::Mat& img, Node& n, Patch& p);


	// member data
	bool initFlag;

	// the image mat for the input image file
	cv::Mat image;

	// the target region
	Patch maskPatch;

	// the raw image data for the input image file
	float *imageData;

	// the raw image data for the input image file on GPU
	float *deviceImageData;
	int imgWidth;
	int imgHeight;

	int patchListSize;
	// the patches list
	Patch *patchList;
	Patch *devicePatchList;

	// the on-GPU SSD table
	SSDEntry *deviceSSDTable;

	int *choiceList;
	// the final resuilt for each node
	int *deviceChoiceList;
	
	int nodeWidth;
	int nodeHeight;
	Node *nodeTable;
	// the on-GPU node table
	Node *deviceNodeTable;

	// the on-GPU message table 
	// we have two chunks of memory for the message table 
	// these two chunks of memory makes the iteration easier
	float *deviceMsgTable;
	float *deviceFillMsgTable;

	// the cost table for the edge region
	float *deviceEdgeCostTable;
};



#endif // __INPAINTING_CUDA_H__
