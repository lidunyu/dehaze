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

Mat DeHaze::guildFilter(Mat& I, Mat& p, int r, double eps)
{
	/*
	% GUIDEDFILTER   O(1) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	Mat _I;
	I.convertTo(_I, CV_64FC1);
	I = _I;

	Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.  
	Mat N;
	boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));

	//mean_I = boxfilter(I, r) ./ N;  
	Mat mean_I;
	boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;  
	Mat mean_p;
	boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;  
	Mat mean_Ip;
	boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.  
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;  
	Mat mean_II;
	boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;  
	Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;     
	Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;  
	Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;  
	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	mean_a = mean_a / N;

	//mean_b = boxfilter(b, r) ./ N;  
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;  
	Mat q = mean_a.mul(I) + mean_b;

	return q;
}
Mat DeHaze::getDarkChannel(Mat &src)
{
	CvSize size = cvSize((src).rows, (src).cols);
	Mat temp = Mat(size, CV_8UC1,Scalar(0));
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
	double sum = 0;   //���ص��������A�ĺ�
	int pointNum = 0;   //����Ҫ������ص���
	double A;        //������ǿA
	double pix;    //��ͨ��ͼ�������ȵ�ǰ0.1%��Χ������ֵ
	//uchar** pixel1;
	//uchar** pixel2;//��ͼ�з���A�ĵ㣬����ͼ�ж�Ӧ������,����ͨ����p1��p2��p3
	//uchar** pixel3;

	float stretch_p[256], stretch_p1[256], stretch_num[256];
	//�����������,��ʼ���������Ԫ��Ϊ0    
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
	//ͳ�Ƹ����Ҷȼ����ֵĸ���  
	for (i = 0; i<256; i++)
	{
		stretch_p[i] = stretch_num[i] / (nHeight*nWidth);
	}

	//ͳ�Ƹ����Ҷȼ��ĸ���,�Ӱ�ͨ��ͼ�а������ȵĴ�Сȡǰ0.1%������,pixΪ�ֽ��
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

Mat DeHaze::getMinIcy(Mat& dark, int w)
{
	CvSize size = cvSize((dark).rows, (dark).cols);
	Mat Icy = Mat(size, CV_8UC1,Scalar(0));
	int hei = dark.rows;
	int wid = dark.cols;
	int hw = hei / w;
	int ww = wid / w;
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		for (int j = w; j < (ww - 1)*w; j += w)
		{
			uchar p =dark.ptr<uchar>(i - 1)[j - 1];  //�õ����������½ǵ�һ�����ص�
														 //�õ�������С������ֵ
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
			//����Icy��ֵ
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

	//�������ұ�һ��  ����������һ���ӿ�
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		uchar p = dark.ptr<uchar>(i - 1)[wid - 1];  //�õ����������½ǵ�һ�����ص�
		for (int ii = i - w; ii < i; ii++)
		{
			for (int j = (ww - 1)*w; j < wid; j++)
			{
				//�õ�������С������ֵ
				uchar newp = dark.ptr<uchar>(ii)[j];
				if (newp < p)
				{
					p = newp;
					Icy.ptr<uchar>(ii)[j] = p;
				}
			}
		}
/*
		//����Icy��ֵ
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				Icy.ptr<uchar>(ii)[j] = p;
			}
		}
*/
	}


	//��������һ�� ���������һ���ӿ�
	for (int j = w; j < (ww - 1)*w; j += w)
	{
		uchar p = dark.ptr<char>(hei - 1)[j];  //�õ����������½ǵ�һ�����ص�
		for (int i = (hw - 1)*w; i < hei; i++)
		{
			for (int jj = j - w; jj < j; jj++)
			{
				//�õ�������С������ֵ
				uchar newp = dark.ptr<uchar>(i)[jj];
				if (newp < p)
				{
					p = newp;
					Icy.ptr<uchar>(i)[jj] = p;
				}
			}
		}
/*
		//����Icy��ֵ
		for (int i = (hw - 1)*w; i < hei; i++)
		{

			for (int jj = j - w; jj < j; jj++)
			{
				Icy.ptr<uchar>(i)[jj] = p;
			}
		}
*/
	}

	//���������½ǵ�һ���ӿ�
	uchar p = dark.ptr<uchar>(hei - 1)[wid - 1];  //�õ����������½ǵ�һ�����ص�
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			//�õ�������С������ֵ
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

Mat DeHaze::getTransmission(Mat& Icy, double Ac)
{
	CvSize size = cvSize((Icy).cols, (Icy).rows);
	Mat t = Mat(size, CV_8UC1,Scalar(0));
	cout << "t rows " << t.rows << " t cols " << t.cols <<" t channals "<<t.channels()<<endl;
	for (int i = 0; i < t.rows; i++)
	{
		for (int j = 0; j < t.cols; j++)
		{
			uchar temp = Icy.ptr<uchar>(i)[j];
			uchar tempt = (uchar)(1 - 0.95*temp / Ac);
			t.ptr<uchar>(i)[j] = temp * 255;
		}
	}
	return t;
}

//convert image depth to CV_64F
Mat DeHaze::getimage(Mat &a)
{
	CvSize size = cvSize(a.rows, a.cols);
	//int hei = a.rows;
	//int wid = a.cols;
	Mat I(size, CV_64FC1);
	//convert image depth to CV_64F  
	a.convertTo(I, CV_64FC1, 1.0 / 255.0);
	cout << "I rows " << I.rows << " I cols " << I.cols << " I channels " << I.channels()<< endl;
	return I;

}

Mat DeHaze::getDehazedImage(Mat hazeImage, IplImage* guidedt, double Ac)
{
	CvSize size = cvSize((hazeImage).rows, (hazeImage).cols);
	Mat dehazedImage = Mat(size, CV_8UC3,Scalar(0,0,0));
	Mat r = Mat(size, CV_8UC1, Scalar(0));
	Mat g = Mat(size, CV_8UC1, Scalar(0));
	Mat b = Mat(size, CV_8UC1, Scalar(0));

	cvSplit(&hazeImage, &b, &g, &r, NULL);

	Mat dehaze_r = Mat(size, CV_8UC1, Scalar(0));
	Mat dehaze_g = Mat(size, CV_8UC1, Scalar(0));
	Mat dehaze_b = Mat(size, CV_8UC1, Scalar(0));

	for (int i = 0; i < r.rows; i++)
	{
		for (int j = 0; j < r.cols; j++)
		{
			double tempt = cvGetReal2D(&guidedt, i, j);
			if (tempt / 255 < 0.1)
			{
				tempt = 25.5;
			}

			uchar I_r = r.ptr<uchar>(i)[j];
			uchar de_r = (uchar)(255 * (I_r - Ac) / tempt + Ac);
			dehaze_r.ptr<uchar>(i)[j] = de_r;

			uchar I_g = g.ptr<uchar>(i)[j];
			uchar de_g = 255 * (I_g - Ac) / tempt + Ac;
			dehaze_g.ptr<uchar>(i)[j] = de_g;
			uchar I_b = b.ptr<uchar>(i)[j];
			uchar de_b = 255 * (I_b - Ac) / tempt + Ac;
			dehaze_b.ptr<uchar>(i)[j] = de_b;
		}
	}

	cvMerge(&dehaze_b, &dehaze_g, &dehaze_r, 0, &dehazedImage);
	return dehazedImage;
}
