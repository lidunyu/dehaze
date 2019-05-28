#pragma once
#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"
#include<vector>
#include "opencv2/core/core_c.h"
#include "opencv2/core/cvstd.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;

class DeHaze
{
public:
	DeHaze();
	void Deinit(IplImage* res);
	~DeHaze();
	static DeHaze* getInstance();
	Mat getimage(Mat &a);
	Mat guildFilter(Mat& I, Mat& p, int r, double eps);
	Mat getDarkChannel(Mat &src);
	Mat getMinIcy(Mat& dark, int w);
	double getA(Mat dark, Mat hazeImage);
	Mat getTransmission(Mat& Icy, double Ac);
	Mat getDehazedImage(Mat hazeImage, IplImage* guidedt, double Ac);
private:
	bool m_init;
	bool is_preview;
	bool is_video;
	IplImage* res;
};
