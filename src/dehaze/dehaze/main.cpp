#include<iostream>
#include "Darkdehaze.h"
int main()
{
	cout << "print here" << endl;
	Mat image= imread("D:\\imageal\\dehaze\\data\\1.jpg");
	CvSize size = cvSize((image).rows, (image).cols);
	Mat g = Mat(size, CV_8UC1,Scalar(0));
	g = DeHaze::getInstance()->getDarkChannel(image);
	double A = DeHaze::getInstance()->getA(g, image);   //大气光强A
	CvSize size1 = cvSize(g.rows, g.cols);
	cout << "print here before Icy" << endl;
	Mat Icy = Mat(size1, CV_8UC1,Scalar(0));
	//imshow("原图", Icy);
	//waitKey(0);
	cout << "print here after Icy" << endl;
	Icy = DeHaze::getInstance()->getMinIcy(g, 5);
	cout << "print here after111 Icy" << endl;
	//投射图t
	Mat t = Mat(size, CV_8UC1,Scalar(0));
	cout << "print here before trans" << endl;
	t = DeHaze::getInstance()->getTransmission(Icy, A);
	cout << "print here after trans" << endl;
	//获得guide image
	Mat image_src = image.clone();
	Mat image_gray(image_src.size(), CV_8UC1,Scalar(0));
	cvtColor(image_src, image_gray, CV_BGR2GRAY);
	Mat guide = DeHaze::getInstance()->getimage(image_gray);
	int r = 8;
	double eps = 0.04;
	cout << "aaaaaa" << endl;
	Mat q = DeHaze::getInstance()->guildFilter(guide, t, r, eps);
	cout << "bbbbbb" << endl;
	IplImage* guidedt = cvCloneImage(&(IplImage)q);
	Mat dehazedImage = Mat(size, CV_8UC3, Scalar(0,0,0));
	dehazedImage = DeHaze::getInstance()->getDehazedImage(image, guidedt, A);
	cout << "print here" << endl;
	imshow("原图", image);
	imshow("去雾后的图", dehazedImage);
	imwrite("D:\\imageal\\dehaze\\datadark.jpg", g);
	imwrite("D:\imageal\dehaze\data\\t.jpg", t);
	imwrite("D:\imageal\dehaze\data\\dehazedImage84.jpg", dehazedImage);
	DeHaze::getInstance()->Deinit(guidedt);
	waitKey(0);
	return 0;
}