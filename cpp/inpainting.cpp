#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


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

patch roundUpArea(patch p) {
	patch res;
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT;
	res.width = (p.x + p.width + NODE_WIDTH - 1) / NODE_WIDTH * NODE_WIDTH - res.x;
	res.height = (p.y + p.height + NODE_WIDTH - 1) / NODE_HEIGHT * NODE_HEIGHT - res.y;
	/*
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH - NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT - NODE_HEIGHT;
	res.width = (p.x + p.width) / NODE_WIDTH * NODE_WIDTH + NODE_WIDTH - res.x;
	res.height = (p.y + p.height) / NODE_HEIGHT * NODE_HEIGHT + NODE_HEIGHT - res.y;
	*/
	return res;
}

bool overlapPatch(patch &p1, patch &p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
	    mRX = (p1.x+p1.width) < (p2.x+p2.width) ? (p1.x+p1.width) : (p2.x+p2.width),
	    mTY = p1.y < p2.y ? p2.y : p1.y,
	    mBY = (p1.y+p1.height) < (p2.y+p2.height) ? (p1.y+p1.height) : (p2.y+p2.height);
	return mRX > mLX && mBY > mTY;
}

vector<patch> genPatches(Mat &img, patch p) {
	vector<patch> res;
	int hh = img.rows / NODE_HEIGHT,
	    ww = img.cols / NODE_WIDTH;
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
			if(img.rows - cY < PATCH_HEIGHT || img.cols - cX < PATCH_WIDTH)
				continue;
			patch cur(cX, cY, PATCH_WIDTH, PATCH_HEIGHT);
			if(!overlapPatch(cur, p))
				res.push_back(cur);
		}
	}
	return res;
}

float calculateSSD(Mat &img, patch &p1, patch &p2, EPOS pos) {
	float res = 0;
	int ww, hh;
	switch(pos) {
		case UP_DOWN:
			/*
			ww = p1.width < p2.width ? p1.width : p2.width;
			hh = (p1.height - NODE_HEIGHT) < p2.height ? (p1.height - NODE_HEIGHT): p2.height;
			*/
			ww = PATCH_WIDTH;
			hh = NODE_HEIGHT;
			for(int i = 0; i < hh; i++) {
				for(int j = 0; j < ww; j++) {
					Vec3f pv1 = img.at<Vec3f>(p1.y + NODE_HEIGHT + i, p1.x + j),
					      pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
					for(int k = 0; k < 3; k++) {
						float m = pv1[k] - pv2[k];
						res += m * m;
					}
				}
			}
			break;
		case LEFT_RIGHT:
			/*
			hh = p1.height < p2.height ? p1.height : p2.height;
			ww = (p1.width - NODE_WIDTH) < p2.width ? (p1.width - NODE_WIDTH) : p2.width;
			*/
			ww = NODE_WIDTH;
			hh = PATCH_HEIGHT;
			for(int i = 0; i < hh; i++) {
				for(int j = 0; j < ww; j++) {
					Vec3f pv1 = img.at<Vec3f>(p1.y + i, p1.x + NODE_WIDTH + j),
					      pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
					for(int k = 0; k < 3; k++) {
						float m = pv1[k] - pv2[k];
						res += m * m;
					}
				}
			}
			break;
		default:
			cout<<"FATAL ERROR"<<endl;
			exit(-1);
	}
	return res;
}

vector<vector<vector<float> > > calculateSSDTable(Mat &img, vector<patch> &patchList) {
	vector<vector<vector<float> > > res;
	int len = patchList.size();
	res.resize(len);
	for(int i = 0; i < len; i++) {
		res[i].resize(len);
	}
	for(int i = 0; i < len; i++) {
		for(int j = i; j < len; j++) {
			if(0/*i == j*/) {
				/*
				 * do nothing since the result must be zero
				 */
				/*
				for(int k = 0; k < EPOS_COUNT; k++) 
					res[i][j].push_back(0);
				*/
			} else {
				res[i][j].resize(EPOS_COUNT);
				res[i][j][UP_DOWN] = calculateSSD(img, patchList[i], patchList[j], UP_DOWN);
				res[i][j][DOWN_UP] = calculateSSD(img, patchList[j], patchList[i], UP_DOWN);
				res[i][j][LEFT_RIGHT] = calculateSSD(img, patchList[i], patchList[j], LEFT_RIGHT);
				res[i][j][RIGHT_LEFT] = calculateSSD(img, patchList[j], patchList[i], LEFT_RIGHT);
				if(res[i][j][UP_DOWN] > FULL_MSG)
					FULL_MSG = res[i][j][UP_DOWN];
				if(res[i][j][DOWN_UP] > FULL_MSG)
					FULL_MSG = res[i][j][DOWN_UP];
				if(res[i][j][LEFT_RIGHT] > FULL_MSG)
					FULL_MSG = res[i][j][LEFT_RIGHT];
				if(res[i][j][RIGHT_LEFT] > FULL_MSG)
					FULL_MSG = res[i][j][RIGHT_LEFT];
			}
		}
	}
	return res;
}

float getSSD(vector<vector<vector<float> > > &ssdTable, int p1, int p2, EPOS pos) {
	if(p1 > p2) {
		switch(pos) {
			case UP_DOWN:
				pos = DOWN_UP;
				break;
			case DOWN_UP:
				pos = UP_DOWN;
				break;
			case LEFT_RIGHT:
				pos = RIGHT_LEFT;
				break;
			case RIGHT_LEFT:
				pos = LEFT_RIGHT;
				break;
			default:
				cout<<"ERROR"<<endl;
				exit(-1);
		}
		return ssdTable[p2][p1][pos];
	}
	return ssdTable[p1][p2][pos];
}

enum EDIR {
	DIR_UP = 0,
	DIR_DOWN,
	DIR_LEFT,
	DIR_RIGHT,
	DIR_COUNT,
};

class node {
	public:
		vector<vector<float> > msg;
		vector<vector<float> > newMsg;
		vector<float> edge_cost;
		int label;
		int x;
		int y;
		bool visted;
};


void initNodeTable(Mat &img, vector<vector<node> > &nodeTable, patch &p, vector<patch> &patchList) {
	int hh = p.height / NODE_HEIGHT + 1,
	    ww = p.width / NODE_WIDTH + 1,
	    len = patchList.size();
	nodeTable.resize(hh);
	cout<<"hh="<<hh<<" ww="<<ww<<endl;
	for(int i = 0; i < hh; i++) {
		nodeTable[i].resize(ww);
		for(int j = 0; j < ww; j++) {
			nodeTable[i][j].msg.resize(DIR_COUNT);
			for(int k = 0; k < DIR_COUNT; k++) {
				nodeTable[i][j].msg[k].resize(len);
				for(int l = 0; l < len; l++) {
					//nodeTable[i][j].msg[k][l] = -1;
					nodeTable[i][j].msg[k][l] = CONST_FULL_MSG;
				}
			}
			nodeTable[i][j].label = -1;
			nodeTable[i][j].x = p.x + j * NODE_WIDTH;
			nodeTable[i][j].y = p.y + i * NODE_HEIGHT;
			//printf("(%d,%d) => (%d,%d)\n", j, i, nodeTable[i][j].x, nodeTable[i][j].y);
			nodeTable[i][j].edge_cost.resize(len);
			for(int k = 0; k < len; k++) {
				float val = 0;
				patch curPatch(0, 0, PATCH_WIDTH, PATCH_HEIGHT);
				//if((i == 0 || i == hh - 1) && (j == 0 || j == ww - 1)) {
				if(((i == 0 || i == hh - 1) && (j >= 0 && j <= ww - 1)) ||
						((j == 0 || j == ww - 1) && (i >= 0 && i <= hh -1))) {
					if(j == 0) {
						curPatch.x = nodeTable[i][j].x - PATCH_WIDTH;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], LEFT_RIGHT);
					} else {
						curPatch.x = nodeTable[i][j].x;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, patchList[k], curPatch, LEFT_RIGHT);
					}
					if(i == 0) {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y - PATCH_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], UP_DOWN);
					} else {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y;
						val += calculateSSD(img, patchList[k], curPatch, UP_DOWN);
					}
				}
				nodeTable[i][j].edge_cost[k] = val;
				if(val < 1)
					nodeTable[i][j].edge_cost[k] = CONST_FULL_MSG;
			}
			nodeTable[i][j].newMsg = nodeTable[i][j].msg;
		}
	}
}

void propagateMsg(vector<vector<node> > &nodeTable, vector<vector<vector<float> > > &ssdTable) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size(),
	    len = ssdTable.size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			for(int k = 0; k < len; k++) {
				float aroundMsg = 0, msgCount, matchFactor;
				float msgFactor = 0.6;
				matchFactor = 10;
				msgCount = msgFactor * 3 + matchFactor;
				//printf("1 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				if(i != 0) {
					aroundMsg += nodeTable[i-1][j].msg[DIR_DOWN][k];
				//printf("11 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
					//msgCount++;
				} else {
					aroundMsg += CONST_FULL_MSG;
				//printf("12 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				}
				//printf("2 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				if(i != hh - 1) {
					aroundMsg += nodeTable[i+1][j].msg[DIR_UP][k];
					//msgCount++;
				} else {
					aroundMsg += CONST_FULL_MSG;
				}
				//printf("3 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				if(j != 0) {
					aroundMsg += nodeTable[i][j-1].msg[DIR_RIGHT][k];
					//msgCount++;
				} else {
					aroundMsg += CONST_FULL_MSG;
				}
				//printf("4 (%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				if(j != ww - 1) {
					aroundMsg += nodeTable[i][j+1].msg[DIR_LEFT][k];
					//msgCount++;
				} else {
					aroundMsg += CONST_FULL_MSG;
				}
				/*
				if(msgCount > 0.5) {
					aroundMsg /= msgCount * 1;
				}
				*/
				aroundMsg *= msgFactor;
				aroundMsg += nodeTable[i][j].edge_cost[k];
				if(nodeTable[i][j].edge_cost[k] > 0.5)
					msgCount++;
				//printf("(%d,%d,%d) => aroundMsg=%f\n", j, i, k, aroundMsg);
				for(int ll = 0; ll < len; ll++) {
					float val, oldVal;
					// up
					if(i != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, DOWN_UP) * matchFactor;
						val -= nodeTable[i-1][j].msg[DIR_DOWN][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_UP][ll];
						if(val < oldVal) {
							nodeTable[i][j].newMsg[DIR_UP][ll] = val;
							/*
							if(i == 1 && j == 1)
							cout<<"less => "<<i<<","<<j<<","<<k<<","<<ll<<" val="<<val<<" oldVal="<<oldVal<<"   last="<<nodeTable[i][j].newMsg[DIR_UP][ll]<<endl;
							*/
						} else {
							nodeTable[i][j].newMsg[DIR_UP][ll] = oldVal;
						}
					}
					// down
					if(i != hh - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, UP_DOWN) * matchFactor;
						val -= nodeTable[i+1][j].msg[DIR_UP][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_DOWN][ll];
						if(val < oldVal) {
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = val;
						} else {
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = oldVal;
						}
					}
					// left
					if(j != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, RIGHT_LEFT) * matchFactor;
						val -= nodeTable[i][j-1].msg[DIR_RIGHT][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_LEFT][ll];
						if(val < oldVal) {
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = val;
						} else {
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = oldVal;
						}
					}
					// right
					if(j != ww - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, LEFT_RIGHT) * matchFactor;
						val -= nodeTable[i][j+1].msg[DIR_LEFT][k] * msgFactor;
						//printf("(%d,%d,-%d) => val=%f\n", j, i, ll, val);
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_RIGHT][ll];
						if(val < oldVal) {
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = val;
						} else {
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = oldVal;
						}
					}
				}
			}
		}
	}
	/*
	int xx = 0, yy = 1;
	printf("(%d,%d) %d\n", xx, yy, len);
	for(int k = 0; k < len; k++) {
		printf("%f ", nodeTable[yy][xx].newMsg[DIR_RIGHT][k]);
	}
	printf("\n");
	*/
	/*
	cout<<"value test old-value="<<nodeTable[1][1].msg[DIR_UP][2]<<endl;
	cout<<"value test old-value="<<nodeTable[1][1].newMsg[DIR_UP][2]<<endl;
	*/
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			nodeTable[i][j].msg = nodeTable[i][j].newMsg;
		}
	}
	/*
	cout<<"value test new-value="<<nodeTable[1][1].msg[DIR_UP][2]<<endl;
	*/
}

void selectPatch(vector<vector<node> > &nodeTable) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			int len = nodeTable[i][j].edge_cost.size();
			float maxB = 0;
			int maxIdx = -1;
			for(int k = 0; k < len; k++) {
				float bl = -nodeTable[i][j].edge_cost[k];
				if(i - 1 >= 0 && nodeTable[i-1][j].msg[DIR_DOWN][k] > 0)
					bl -= nodeTable[i-1][j].msg[DIR_DOWN][k];
				if(i + 1 < hh && nodeTable[i+1][j].msg[DIR_UP][k] > 0)
					bl -= nodeTable[i+1][j].msg[DIR_UP][k];
				if(j - 1 >= 0 && nodeTable[i][j-1].msg[DIR_RIGHT][k] > 0) 
					bl -= nodeTable[i][j-1].msg[DIR_RIGHT][k];
				if(j + 1 < ww && nodeTable[i][j+1].msg[DIR_LEFT][k] > 0) 
					bl -= nodeTable[i][j+1].msg[DIR_LEFT][k];
				if(i == 1 && j == 1) {
					//cout<<k<<" =>"<<bl<<","<<FULL_MSG<<endl;
				}
				if(bl > maxB || maxIdx < 0) {
					maxB = bl;
					maxIdx = k;
				}
			}
			cout<<i<<","<<j<<" => "<<maxB<<" "<<maxIdx<<endl;
			nodeTable[i][j].label = maxIdx;
		}
	}
}

void pastePatch(Mat &img, node &n, patch &p) {
	int xx = n.x - NODE_WIDTH,
	    yy = n.y - NODE_HEIGHT;
	for(int i = 0; i < p.height; i++) {
		for(int j = 0; j < p.width; j++) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + i, p.x + j);
			//img.at<Vec3f>(yy + i, xx + j) = Vec3f(0, 0, 0);
		}
	}
}

void fillPatch(Mat &img, vector<vector<node> > &nodeTable, vector<patch> &patchList) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			cout<<nodeTable[i][j].label<<" ";
		}
		cout<<endl;
	}
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			int label = nodeTable[i][j].label;
			if(label >= 0) {
				pastePatch(img, nodeTable[i][j], patchList[label]);
			}
		}
	}
}

#define RR 1
void drawPoint(Mat &img, int x, int y) {
	for(int i = x - RR; i < x + RR; i++) {
		for(int j = y - RR; j < y + RR; j++) {
			img.at<Vec3f>(j, i) = Vec3f(0.0f, 0.0f, 0.0f);
		}
	}
}

void drawRect(Mat &img, patch &p) {
	for(int i =  p.y; i < p.y + p.height; i++) {
		for(int j = p.x; j < p.x + p.width; j++) {
			if((i == p.y) || (i == p.y + p.height - 1) || (j == p.x) || (j == p.x + p.width - 1))
				drawPoint(img, j, i);
		}
	}
}

int main(int argc, char **argv) {
	if(argc != 8) {
		cout<<"Usage: "<<argv[0]<<" input x y w h output iter_time"<<endl;
		return 0;
	}
	char *input = argv[1],
	     *output = argv[6];
	int maskX = atoi(argv[2]),
	    maskY = atoi(argv[3]),
	    maskW = atoi(argv[4]),
	    maskH = atoi(argv[5]),
	    iterTime = atoi(argv[7]);
	Mat img;
	img = imread(input, CV_LOAD_IMAGE_COLOR);
	img.convertTo(img, CV_32FC3);

	if(!img.data) {
		cout<<"Load Image failed"<<endl;
		return 0;
	}
	cout<<"W="<<img.cols<<endl;
	cout<<"H="<<img.rows<<endl;

	patch missing = roundUpArea(patch(maskX, maskY, maskW, maskH));
	cout<<"x="<<missing.x<<" y="<<missing.y<<" width="<<missing.width<<" height="<<missing.height<<endl;
	vector<patch> patchList = genPatches(img, missing);
	cout<<"Patch Size: "<<patchList.size()<<endl;
	cout<<"patch => " << 2 << " : x=" << patchList[2].x << " y=" << patchList[2].y << endl;
	vector<vector<vector<float> > > ssdTable = calculateSSDTable(img, patchList);

	cout << "SSD 2=>7 UP_DOWN:" << getSSD(ssdTable, 2, 7, LEFT_RIGHT) << endl;

	vector<vector<node> > nodeTable;
	initNodeTable(img, nodeTable, missing, patchList);
	/*
	for(int i = 0; i < nodeTable.size(); i++) {
		for(int j = 0; j < nodeTable.size(); j++) {
			cout << "(" << j <<","<<i<<") ";
			for(int k = 0; k < nodeTable[i][j].edge_cost.size(); k++) {
				cout << nodeTable[i][j].edge_cost[k] << " ";
			}
			cout << endl;
		}
	}
	*/
	
	//fillPatch(img, nodeTable);
	for(int i = 0; i < iterTime; i++) {
	//for(int i = 0; i < 1; i++) {
		propagateMsg(nodeTable, ssdTable);
		cout<<"ITERATION "<<i<<endl;
	}
	selectPatch(nodeTable);
	fillPatch(img, nodeTable, patchList);
	drawRect(img, missing);

	imwrite(output, img);
	return 0;
}
