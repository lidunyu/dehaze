#include "Darkdehaze.h"
#include<iostream>
#include<vector>
DeHaze::DeHaze()
	: m_init(false)
	, is_preview(false)
	, is_video(false)
	, res(NULL)
{}
DeHaze::~DeHaze()
{
	if (!res)
		cvReleaseImage(&res);
	res = NULL;
}

void DeHaze::Deinit(IplImage *res)
{
	DeHaze::~DeHaze();
}

DeHaze* DeHaze::getInstance()
{
	static DeHaze res;
	return &res;
}

Mat DeHaze::guildFilter(Mat I, Mat p, int r, double eps)
{
	/*
	% GUIDEDFILTER   O(1) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_64FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.  
	cv::Mat N;
	cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));

	//mean_I = boxfilter(I, r) ./ N;  
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;  
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;  
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.  
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;  
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;  
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;     
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;  
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;  
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	mean_a = mean_a / N;

	//mean_b = boxfilter(b, r) ./ N;  
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;  
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}
Mat DeHaze::getDarkChannel(Mat &src)
{
	CvSize size = cvSize((src).rows, (src).cols);
	Mat temp = Mat(size, IPL_DEPTH_8U, 1);
	CvScalar pixel;
	double  px;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			pixel = cvGet2D(&src, i, j);
			if (pixel.val[0]<pixel.val[1])
			{
				px = pixel.val[0];
			}
			else
			{
				px = pixel.val[1];
			}

			if (px >pixel.val[2])
			{
				px = pixel.val[2];
			}
			
			cvSetReal2D(&temp, i, j, px);
		}
	}
	return  temp;
}

double DeHaze::getA(Mat dark, Mat hazeImage)
{
	double sum = 0;   //像素点符合条件A的和
	int pointNum = 0;   //满足要求的像素点数
	double A;        //大气光强A
	double pix;    //暗通道图中照亮度的前0.1%范围的像素值
	CvScalar pixel;   //按图中符合A的点，在雾图中对应的像素

	float stretch_p[256], stretch_p1[256], stretch_num[256];
	//清空三个数组,初始化填充数组元素为0    
	memset(stretch_p, 0, sizeof(stretch_p));
	memset(stretch_p1, 0, sizeof(stretch_p1));
	memset(stretch_num, 0, sizeof(stretch_num));

	int nHeight = dark.rows;
	int nWidth = dark.cols;
	int i, j;
	for (i = 0; i<nHeight; i++)
	{
		for (j = 0; j<nWidth; j++)
		{
			double  pixel0 = cvGetReal2D(&dark, i, j);
			int   pixel = (int)pixel0;
			stretch_num[pixel]++;
		}
	}
	//统计各个灰度级出现的概率  
	for (i = 0; i<256; i++)
	{
		stretch_p[i] = stretch_num[i] / (nHeight*nWidth);
	}

	//统计各个灰度级的概率,从暗通道图中按照亮度的大小取前0.1%的像素,pix为分界点
	for (i = 0; i<256; i++)
	{
		for (j = 0; j <= i; j++)
		{
			stretch_p1[i] += stretch_p[j];
			if (stretch_p1[i]>0.999)
			{
				pix = (double)i;
				i = 256;
				break;
			}

		}
	}

	for (i = 0; i< hazeImage.rows; i++)
	{
		for (j = 0; j < hazeImage.cols; j++)
		{
			double temp = cvGetReal2D(&dark, i, j);
			if (temp > pix)
			{
				pixel = cvGet2D(&hazeImage, i, j);
				pointNum++;
				sum += pixel.val[0];
				sum += pixel.val[1];
				sum += pixel.val[2];

			}
		}
	}
	A = sum / (3 * pointNum);
	if (A > 220.0)
	{
		A = 220.0;
	}
	return A;
}

Mat DeHaze::getMinIcy(Mat dark, int w)
{
	Mat Icy = Mat(cvGetSize(&dark), IPL_DEPTH_8U, 1);
	int hei = dark.rows;
	int wid = dark.cols;
	int hw = hei / w;
	int ww = wid / w;
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		for (int j = w; j < (ww - 1)*w; j += w)
		{
			double p = cvGetReal2D(&dark, i - 1, j - 1);  //得到窗口最右下角的一个像素点
														 //得到窗口最小的像素值
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					double newp = cvGetReal2D(&dark, ii, jj);
					if (newp < p)
					{
						p = newp;
					}
				}
			}
			//设置Icy的值
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					cvSetReal2D(&Icy, ii, jj, p);
				}
			}

		}
	}

	//处理最右边一列  不包含最下一个子块
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		double p = cvGetReal2D(&dark, i - 1, wid - 1);  //得到窗口最右下角的一个像素点
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				//得到窗口最小的像素值
				double newp = cvGetReal2D(&dark, ii, j);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//设置Icy的值
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				cvSetReal2D(&Icy, ii, j, p);
			}
		}
	}


	//处理最下一行 不包含最后一个子块
	for (int j = w; j < (ww - 1)*w; j += w)
	{
		double p = cvGetReal2D(&dark, hei - 1, j);  //得到窗口最右下角的一个像素点
		for (int i = (hw - 1)*w; i < hei; i++)
		{
			for (int jj = j - w; jj < j; jj++)
			{
				//得到窗口最小的像素值
				double newp = cvGetReal2D(&dark, i, jj);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//设置Icy的值
		for (int i = (hw - 1)*w; i < hei; i++)
		{

			for (int jj = j - w; jj < j; jj++)
			{
				cvSetReal2D(&Icy, i, jj, p);
			}
		}

	}

	//处理最右下角的一个子块
	double p = cvGetReal2D(&dark, hei - 1, wid - 1);  //得到窗口最右下角的一个像素点
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			//得到窗口最小的像素值
			double newp = cvGetReal2D(&dark, i, j);
			if (newp < p)
			{
				p = newp;
			}

		}
	}
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			cvSetReal2D(&Icy, i, j, p);

		}
	}

	return Icy;

}

Mat DeHaze::getTransmission(Mat Icy, double Ac)
{
	Mat t = Mat(cvGetSize(&Icy), IPL_DEPTH_8U, 1);
	for (int i = 0; i < t.rows; i++)
	{
		for (int j = 0; j < t.cols; j++)
		{
			double temp = cvGetReal2D(&Icy, i, j);
			double tempt = 1 - 0.95*temp / Ac;
			cvSetReal2D(&t, i, j, tempt * 255);
		}
	}

	return t;
}

//convert image depth to CV_64F
Mat DeHaze::getimage(Mat &a)
{
	int hei = a.rows;
	int wid = a.cols;
	Mat I(hei, wid, CV_64FC1);
	//convert image depth to CV_64F  
	a.convertTo(I, CV_64FC1, 1.0 / 255.0);
	return I;

}

Mat DeHaze::getDehazedImage(Mat hazeImage, IplImage* guidedt, double Ac)
{
	Mat dehazedImage = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 3);
	Mat r = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);
	Mat g = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);
	Mat b = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);

	cvSplit(&hazeImage, &b, &g, &r, NULL);

	Mat dehaze_r = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);
	Mat dehaze_g = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);
	Mat dehaze_b = Mat(cvGetSize(&hazeImage), IPL_DEPTH_8U, 1);

	for (int i = 0; i < r.rows; i++)
	{
		for (int j = 0; j < r.cols; j++)
		{
			double tempt = cvGetReal2D(&guidedt, i, j);
			if (tempt / 255 < 0.1)
			{
				tempt = 25.5;
			}

			double I_r = cvGetReal2D(&r, i, j);
			double de_r = 255 * (I_r - Ac) / tempt + Ac;
			cvSetReal2D(&dehaze_r, i, j, de_r);

			double I_g = cvGetReal2D(&g, i, j);
			double de_g = 255 * (I_g - Ac) / tempt + Ac;
			cvSetReal2D(&dehaze_g, i, j, de_g);

			double I_b = cvGetReal2D(&b, i, j);
			double de_b = 255 * (I_b - Ac) / tempt + Ac;
			cvSetReal2D(&dehaze_b, i, j, de_b);

		}
	}

	cvMerge(&dehaze_b, &dehaze_g, &dehaze_r, 0, &dehazedImage);
	return dehazedImage;
}
