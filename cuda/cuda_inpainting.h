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



class CudaInpainting {
public:
	CudaInpainting(const char *path);
	bool Inpainting(int x, int y, int width, int height, int iterTime);
	cv::Mat GetImage();
	~CudaInpainting();

	enum EPOS {
		UP_DOWN = 0,
		DOWN_UP,
		LEFT_RIGHT,
		RIGHT_LEFT,
		EPOS_COUNT,
	};
	
	enum EDIR {
		DIR_UP = 0, 
		DIR_DOWN,
		DIR_LEFT,
		DIR_RIGHT,
		DIR_COUNT,
	};
	static const int RADIUS;
	static const float RANGE_RATIO;

	static const int PATCH_WIDTH;
	static const int PATCH_HEIGHT;
	static const int NODE_WIDTH;
	static const int NODE_HEIGHT;
	static const float CONST_FULL_MSG;


	class Patch {
	public:
		int x;
		int y;
		int width;
		int height;
		Patch() : x(0), y(0), width(0), height(0) {}
		Patch(int ww, int hh);
		Patch(int xx, int yy, int ww, int hh) : x(xx), y(yy), width(ww), height(hh) {}
	};

	class Node {
	public:
		int x;
		int y;
	};

	class SSDEntry {
	public:
		float data[EPOS_COUNT];
	};

private:
	// member function
	Patch RoundUpArea(Patch p);
	bool OverlapPatch(Patch &p1, Patch &p2);
	void GenPatches();
	float CalculateSSD(Patch& p1, Patch& p2, EPOS pos);
	void CalculateSSDTable();
	void InitNodeTable();
	void RunIteration();
	void SelectPatch();
	void FillPatch();

	static void PastePatch(cv::Mat& img, Node& n, Patch& p);


	// member data
	bool initFlag;

	cv::Mat image;
	Patch maskPatch;

	float *imageData;
	float *deviceImageData;
	int imgWidth;
	int imgHeight;

	int patchListSize;
	Patch *patchList;
	Patch *devicePatchList;

	SSDEntry *deviceSSDTable;

	int *choiceList;
	int *deviceChoiceList;
	
	int nodeWidth;
	int nodeHeight;
	Node *nodeTable;
	Node *deviceNodeTable;
	float *deviceMsgTable;
	float *deviceFillMsgTable;
	float *deviceEdgeCostTable;
};



#endif // __INPAINTING_CUDA_H__
