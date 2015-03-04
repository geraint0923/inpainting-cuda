
#include "cuda_inpainting.h"
#include <cuda.h>
#include <vector>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;


const int CudaInpainting::RADIUS = 32;
const float CudaInpainting::RANGE_RATIO = 2.0f;

const int CudaInpainting::PATCH_WIDTH = CudaInpainting::RADIUS;
const int CudaInpainting::PATCH_HEIGHT = CudaInpainting::RADIUS;
const int CudaInpainting::NODE_WIDTH = CudaInpainting::PATCH_WIDTH / 2;
const int CudaInpainting::NODE_HEIGHT = CudaInpainting::PATCH_HEIGHT / 2;
const float CudaInpainting::CONST_FULL_MSG = CudaInpainting::PATCH_WIDTH * 
			CudaInpainting::PATCH_HEIGHT * 255 * 255 * 3 / 2.0f;

static void CopyToDevice(void *src, void *dst, uint32_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

static void CopyFromDevice(void *src, void *dst, uint32_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// public functions
CudaInpainting::CudaInpainting(const char *path) {
	initFlag = false;
	image = imread(path, CV_LOAD_IMAGE_COLOR);
	imageData = nullptr;
	if(!image.data) {
		cout << "Image loading failed" << endl;
		return;
	}
	image.convertTo(image, CV_32FC3);
	
	// copy the image data to float array
	imageData = new float[3 * image.cols * image.rows];
	cudaMalloc((void**)&deviceImageData, sizeof(float) * 3 * image.cols * image.rows);
	if(!imageData) {
		cout << "Memory allocation failed" << endl;
		cudaFree(deviceImageData);
		return;
	}
	for(int y = 0; y < image.rows; ++y) {
		for(int x = 0; x < image.cols; ++x) {
			Vec3f vec = image.at<Vec3f>(y, x);
			imageData[3 * image.cols * y + 3 * x] = vec[0];
			imageData[3 * image.cols * y + 3 * x + 1] = vec[1];
			imageData[3 * image.cols * y + 3 * x + 2] = vec[2];
		}
	}
	CopyToDevice(imageData, deviceImageData, sizeof(float) * 3 * image.cols * image.rows);

	choiceList = nullptr;
	nodeTable = nullptr;
	patchList = nullptr;
	
	devicePatchList = nullptr;
	deviceSSDTable = nullptr;
	deviceNodeTable = nullptr;
	deviceMsgTable = nullptr;
	deviceEdgeCostTable = nullptr;
	deviceChoiceList = nullptr;
}

CudaInpainting::~CudaInpainting() {
	if(imageData) {
		delete imageData;
		cudaFree(deviceImageData);
	}
	if(choiceList)
		delete choiceList;
	if(patchList)
		delete patchList;
	if(nodeTable)
		delete nodeTable;

	if(devicePatchList)
		cudaFree(devicePatchList);
	if(deviceNodeTable)
		cudaFree(deviceNodeTable);
	if(deviceMsgTable)
		cudaFree(deviceMsgTable);
	if(deviceEdgeCostTable)
		cudaFree(deviceEdgeCostTable);
	if(deviceChoiceList)
		cudaFree(deviceChoiceList);
}

bool CudaInpainting::Inpainting(int x,int y, int width, int height, int iterTime) {
	Patch patch(x, y, width, height);
	maskPatch = RoundUpArea(patch);

	GenPatches();

	CalculateSSDTable();

	InitNodeTable();

	for(int i = 0; i < iterTime; i++) {
		RunIteration();
		cout<<"ITERATION "<<i<<endl;
	}

	SelectPatch();

	FillPatch();

	return true;
}

// private functions
CudaInpainting::Patch CudaInpainting::RoundUpArea(Patch p) {
	Patch res;
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT;
	res.width = (p.x + p.width +NODE_WIDTH - 1) / NODE_WIDTH * NODE_WIDTH - res.x;
	res.height = (p.y + p.height + NODE_WIDTH - 1) / NODE_HEIGHT * NODE_HEIGHT - res.y;
	return res;
}


bool CudaInpainting::OverlapPatch(Patch& p1, Patch& p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
	    mRX = (p1.x+p1.width) < (p2.x+p2.width) ? (p1.x+p1.width) : (p2.x+p2.width),
	    mTY = p1.y < p2.y ? p2.y : p1.y,
	    mBY = (p1.y+p1.height) < (p2.y+p2.height) ? (p1.y+p1.height) : (p2.y+p2.height);
	return mRX > mLX && mBY > mTY;
}

void CudaInpainting::GenPatches() {
	vector<Patch> tmpPatchList;
	Patch p = maskPatch;
	int hh = image.rows / NODE_HEIGHT,
	    ww = image.cols / NODE_WIDTH;
	float midX = p.x + p.width / 2,
	      midY = p.y + p.height / 2;
	for(int i = 1; i <= hh; i++) {
		for(int j = 1; j <= ww; j++) {
			int cX, cY;
			float fcx = j * NODE_WIDTH, fcy = i * NODE_HEIGHT;
			cY = i * NODE_HEIGHT - NODE_HEIGHT;
			cX = j * NODE_WIDTH - NODE_WIDTH;
			if(!(fabsf(fcx - midX) * 2 / p.width < RANGE_RATIO && fabsf(fcy - midY) * 2 / p.height < RANGE_RATIO))
				continue;
			if(image.rows - cY < PATCH_HEIGHT || image.cols - cX < PATCH_WIDTH)
				continue;
			Patch cur(cX, cY, PATCH_WIDTH, PATCH_HEIGHT);
			if(!OverlapPatch(cur, p))
				tmpPatchList.push_back(cur);
		}
	}
	cudaMalloc((void**)&devicePatchList, sizeof(Patch) * tmpPatchList.size());
	patchList = new Patch[tmpPatchList.size()];
	if(!patchList) {
		cout << "NULL patchList! exit"<< endl;
		exit(-1);
	}
	for(int i = 0; i < tmpPatchList.size(); i++) {
		//CopyToDevice(&patchList[i], devicePatchList + i, sizeof(Patch));
		patchList[i] = tmpPatchList[i];
	}
	patchListSize = tmpPatchList.size();
	CopyToDevice(patchList, devicePatchList, sizeof(Patch) * tmpPatchList.size());
}

void CudaInpainting::CalculateSSDTable() {
	cudaMalloc((void**)&deviceSSDTable, sizeof(float) * patchListSize * patchListSize * EPOS_COUNT);
	if(devicePatchList && deviceSSDTable) {
		cout << "Calculate SSDTable" << endl;
	}
}

void CudaInpainting::InitNodeTable() {
}

void CudaInpainting::RunIteration() {
}

__global__ void deviceSelectPatch(float *dEdgeCostTable, float *dChoiceList, int nrPatch) {	
}
void CudaInpainting::SelectPatch() {
	if(choiceList && deviceChoiceList && deviceEdgeCostTable) {
		cout << "Select the Best Patch" << endl;
		deviceSelectPatch<<<dim3(), dim3(1,1)>>>(deviceEdgeCostTable, deviceChoiceList, patchListSize);
		CopyFromDevice(deviceChoiceList, choiceList, sizeof(int) * nodeWidth * nodeHeight);
	}
}

void CudaInpainting::PastePatch(Mat& img, Node& n, Patch& p) {
	int xx = n.x - NODE_WIDTH,
	    yy = n.y - NODE_HEIGHT;
	for(int i = 0; i < p.height; ++i) {
		for(int j = 0; j < p.width; ++j) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + i, p.x + j);
		}
	}
}

void CudaInpainting::FillPatch() {
	int hh = nodeHeight,
	    ww = nodeWidth;
	for(int i = 0; i < hh; ++i) {
		for(int j = 0; j < ww; ++j) {
			cout<<choiceList[j + i * ww]<<" ";
		}
		cout<<endl;
	}
	for(int i = 0; i < hh; ++i) {
		for(int j = 0; j < ww; ++j) {
			int label = choiceList[j + i * ww];
			if(label >= 0) {
				PastePatch(image, nodeTable[j + i * ww], patchList[label]);
			}
		}
	}
}



