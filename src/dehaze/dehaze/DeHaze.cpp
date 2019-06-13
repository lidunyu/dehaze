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

Mat DeHaze::guildFilter(Mat I_org, Mat p_org, int r, double eps, int s)
{
	/*
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat I, _I;
	I_org.convertTo(_I, CV_64FC1, 1.0 / 255);

	resize(_I, I, Size(), 1.0 / s, 1.0 / s, 1);



	cv::Mat p, _p;
	p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
	//p = _p;
	resize(_p, p, Size(), 1.0 / s, 1.0 / s, 1);

	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;

	r = (2 * r + 1) / s + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4   

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
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows), 1);

	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows), 1);

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;

	return q;
}

Mat DeHaze::getDarkChannel(Mat src)
{
	//CvSize size = cvSize((src).rows, (src).cols);
	//int height = src.rows;
	//int wide = src.cols;
	Mat temp = Mat(src.size(), CV_8UC1);
	cout << "temp rows " << temp.rows << " temp cols " << temp.cols << " temp channels " << temp.channels() << endl;
	uchar  px;
	for (int i = 0; i < src.rows; i++)
	{
		uchar* pixel1 = src.ptr<uchar>(i);
		uchar* pixel2 = src.ptr<uchar>(i);
		uchar* pixel3 = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			pixel1[j] = src.ptr<uchar>(i)[j * 3];
			pixel2[j] = src.ptr<uchar>(i)[j * 3+1];
			pixel3[j] = src.ptr<uchar>(i)[j * 3 + 2];
			if (pixel1[j]<pixel2[j])
			{
				px = pixel1[j];
			}
			else
			{
				px = pixel2[j];
			}

			if (px >pixel3[j])
			{
				px = pixel3[j];
			}
			temp.ptr<uchar>(i)[j] = px;
		}
	}
	return  temp;
}

double DeHaze::getA(Mat dark, Mat hazeImage)
{
	double sum = 0;   //像素点符合条件A的和
	int pointNum = 0;   //满足要求的像素点数
	double A;        //大气光强A
	uchar pix;    //暗通道图中照亮度的前0.1%范围的像素值
	//uchar** pixel1;
	//uchar** pixel2;//按图中符合A的点，在雾图中对应的像素,三个通道，p1、p2、p3
	//uchar** pixel3;

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
			uchar  pixel0 = dark.ptr<uchar>(i)[j];
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
				pix = (uchar)i;
				i = 256;
				break;
			}

		}
	}

	for (i = 0; i< hazeImage.rows; i++)
	{
		uchar* pixel1 = hazeImage.ptr<uchar>(i);
		uchar* pixel2 = hazeImage.ptr<uchar>(i);
		uchar* pixel3 = hazeImage.ptr<uchar>(i);
		for (j = 0; j < hazeImage.cols; j++)
		{
			uchar temp = dark.ptr<uchar>(i)[j];
			if (temp > pix)
			{
				pixel1[j] = hazeImage.ptr<uchar>(i)[j * 3];
				pixel2[j] = hazeImage.ptr<uchar>(i)[j * 3 + 1];
				pixel3[j] = hazeImage.ptr<uchar>(i)[j * 3 + 2];
				pointNum++;
				sum += pixel1[j];
				sum += pixel2[j];
				sum += pixel3[j];

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
	//CvSize size = cvSize((dark).rows, (dark).cols);
	Mat Icy = Mat(dark.size(), CV_8UC1,Scalar(0));
	int hei = dark.rows;
	int wid = dark.cols;
	int hw = hei / w;
	int ww = wid / w;
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		for (int j = w; j < (ww - 1)*w; j += w)
		{
			uchar p =dark.ptr<uchar>(i - 1)[j - 1];  //得到窗口最右下角的一个像素点
														 //得到窗口最小的像素值
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					uchar newp = dark.ptr<uchar>(ii)[jj];
					if (newp < p)
					{
						p = newp;
						Icy.ptr<uchar>(ii)[jj] = p;
					}
				}
			}
			//设置Icy的值
/*
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					Icy.ptr<uchar>(ii)[jj] = p;
				}
			}
*/
		}
	}

	//处理最右边一列  不包含最下一个子块
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		uchar p = dark.ptr<uchar>(i - 1)[wid - 1];  //得到窗口最右下角的一个像素点
		for (int ii = i - w; ii < i; ii++)
		{
			for (int j = (ww - 1)*w; j < wid; j++)
			{
				//得到窗口最小的像素值
				uchar newp = dark.ptr<uchar>(ii)[j];
				if (newp < p)
				{
					p = newp;
					Icy.ptr<uchar>(ii)[j] = p;
				}
			}
		}
/*
		//设置Icy的值
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				Icy.ptr<uchar>(ii)[j] = p;
			}
		}
*/
	}


	//处理最下一行 不包含最后一个子块
	for (int j = w; j < (ww - 1)*w; j += w)
	{
		uchar p = dark.ptr<char>(hei - 1)[j];  //得到窗口最右下角的一个像素点
		for (int i = (hw - 1)*w; i < hei; i++)
		{
			for (int jj = j - w; jj < j; jj++)
			{
				//得到窗口最小的像素值
				uchar newp = dark.ptr<uchar>(i)[jj];
				if (newp < p)
				{
					p = newp;
					Icy.ptr<uchar>(i)[jj] = p;
				}
			}
		}
/*
		//设置Icy的值
		for (int i = (hw - 1)*w; i < hei; i++)
		{

			for (int jj = j - w; jj < j; jj++)
			{
				Icy.ptr<uchar>(i)[jj] = p;
			}
		}
*/
	}

	//处理最右下角的一个子块
	uchar p = dark.ptr<uchar>(hei - 1)[wid - 1];  //得到窗口最右下角的一个像素点
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			//得到窗口最小的像素值
			uchar newp = dark.ptr<uchar>(i)[j];
			if (newp < p)
			{
				p = newp;
				Icy.ptr<uchar>(i)[j] = p;
			}

		}
	}
/*
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			Icy.ptr<uchar>(i)[j] = p;
		}
	}
*/
	return Icy;

}

Mat DeHaze::getTransmission(Mat Icy, double Ac)
{
	//CvSize size = cvSize((Icy).rows, (Icy).cols);
	Mat t = Mat(Icy.size(), CV_8UC1,Scalar(0));
	cout << "t rows " << t.rows << " t cols " << t.cols <<" t channals "<<t.channels()<<endl;
	for (int i = 0; i < t.rows; i++)
	{
		for (int j = 0; j < t.cols; j++)
		{
			uchar temp = Icy.ptr<uchar>(i)[j];
			uchar tempt = (uchar)(1 - 0.95*temp / Ac);
			t.ptr<uchar>(i)[j] = tempt * 255;
		}
	}
	return t;
}

//convert image depth to CV_64F
Mat DeHaze::getimage(Mat a)
{
	//CvSize size = cvSize(a.rows, a.cols);
	//int hei = a.rows;
	//int wid = a.cols;
	Mat I(a.size(), CV_64FC1);
	//convert image depth to CV_64F  
	a.convertTo(I, CV_64FC1, 1.0 / 255.0);
	cout << "I rows " << I.rows << " I cols " << I.cols << " I channels " << I.channels()<< endl;
	return I;

}

Mat DeHaze::getDehazedImage(Mat hazeImage, Mat guidedt, double Ac)
{
	//CvSize size = cvSize((hazeImage).rows, (hazeImage).cols);
	//Mat hImage;
	//hazeImage.copyTo(hImage);
	//Mat dehazedImage = Mat(hazeImage.size(), CV_8UC3);
	Mat r, b, g;
	vector<Mat> rgbChannels;
	//vector<Mat> mbgr(3);
	split(hazeImage, rgbChannels);
	b = rgbChannels.at(0);
	g = rgbChannels.at(1);
	r = rgbChannels.at(2);
	//Mat bk1(size, CV_8UC1, Scalar(0));
	//Mat imageB(size, CV_8UC3);
	//mbgr[0] = rgbChannels[0];
	//mbgr[1] = bk1;
	//mbgr[2] = bk1;
	//merge(mbgr, imageB);
	cout << "b rows " << b.rows << " b cols " << b.cols << " b channels " << b.channels() << endl;
	imshow("b", r);
	for (int i = 0; i < r.rows; i++)
	{
		for (int j = 0; j < r.cols; j++)
		{
			uchar tempt = guidedt.ptr<uchar>(i)[j];
			if (tempt / 255 < 0.1)
			{
				tempt = 25.5;
			}
			r.ptr<uchar>(i)[j] = 255 * (r.ptr<uchar>(i)[j] - Ac) / tempt + Ac;
			g.ptr<uchar>(i)[j]= 255 * (g.ptr<uchar>(i)[j] - Ac) / tempt + Ac;
			b.ptr<uchar>(i)[j]= 255 * (b.ptr<uchar>(i)[j] - Ac) / tempt + Ac;
		}
	}
	imshow("r", r);
	merge(rgbChannels, hazeImage);
	cout << "hazeImage rows " << hazeImage.rows << " hazeImage cols " << hazeImage.cols << " hazeImage channels " << hazeImage.channels() << endl;
	imshow("hazeImage", hazeImage);
	return hazeImage;
}
