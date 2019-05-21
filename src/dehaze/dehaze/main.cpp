#include<iostream>
#include "Darkdehaze.h"
int main()
{
	Mat image= imread("D:\\imageal\\dehaze\\data\\1.jpg");
	CvSize size = cvSize((image).rows, (image).cols);
	Mat g = Mat(size, IPL_DEPTH_8U, 1);
	g = DeHaze::getInstance()->getDarkChannel(image);
	double A = DeHaze::getInstance()->getA(g, image);   //大气光强A

	Mat Icy = Mat(size, IPL_DEPTH_8U, 1);
	Icy = DeHaze::getInstance()->getMinIcy(g, 5);

	//投射图t
	Mat t = Mat(size, IPL_DEPTH_8U, 1);
	t = DeHaze::getInstance()->getTransmission(Icy, A);

	//获得guide image
	Mat image_src = image.clone();
	Mat image_gray(image_src.size(), CV_8UC1);
	cvtColor(image_src, image_gray, CV_BGR2GRAY);
	Mat guide = DeHaze::getInstance()->getimage(image_gray);
	int r = 8;
	double eps = 0.04;
	Mat q = DeHaze::getInstance()->guildFilter(guide, t, r, eps);
	IplImage* guidedt = cvCloneImage(&(IplImage)q);
	Mat dehazedImage = Mat(size, IPL_DEPTH_8U, 3);
	dehazedImage = DeHaze::getInstance()->getDehazedImage(image, guidedt, A);
	imwrite("C:\\Users\\徐图之\\Desktop\\dark .jpg", g);
	imwrite("C:\\Users\\徐图之\\Desktop\\t.jpg", t);
	imwrite("C:\\Users\\徐图之\\Desktop\\dehazedImage84.jpg", dehazedImage);

	imshow("原图", image);
	imshow("去雾后的图", dehazedImage);
	DeHaze::getInstance()->Deinit(guidedt);
	waitKey(0);
	return 0;
}