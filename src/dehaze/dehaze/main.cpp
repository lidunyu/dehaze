#include<iostream>
#include "Darkdehaze.h"
int main()
{
	cout << "print here" << endl;
	Mat image= imread("D:\\imageal\\dehaze\\data\\1.jpg");
	imshow("原图", image);
	cout << "image rows " << image.rows << " image cols " << image.cols << " image channels " << image.channels() << endl;
	//CvSize size = cvSize((image).rows, (image).cols);
	Mat g= Mat(image.size(), CV_8UC1,Scalar(0));
	cout << "g rows " << g.rows << " g cols " << g.cols << " g channels " << g.channels() << endl;
	Mat image1;
	image.copyTo(image1);
	imshow("aaa", image1);
	g = DeHaze::getInstance()->getDarkChannel(image1);
	cout << "gg rows " << g.rows << " gg cols " << g.cols << " gg channels " << g.channels() << endl;
	imshow("ccc", image1);
	cout << "ccc rows " << image1.rows << " ccc cols " << image1.cols << " ccc channels " << image1.channels() << endl;
	imshow("g", g);
	double A = DeHaze::getInstance()->getA(g, image1);   //大气光强A
	//CvSize size1 = cvSize(g.rows, g.cols);
	cout << "print here before Icy" << endl;
	Mat Icy = Mat(g.size(), CV_8UC1,Scalar(0));
	cout << "Icy rows " << Icy.rows << " Icy cols " << Icy.cols << " Icy channels " << Icy.channels() << endl;
	//imshow("原图", Icy);
	cout << "print here after Icy" << endl;
	Icy = DeHaze::getInstance()->getMinIcy(g, 5);
	cout << "print here after111 Icy" << endl;
	cout << "Icy rows " << Icy.rows << " Icy cols " << Icy.cols << " Icy channels " << Icy.channels() << endl;
	//投射图t
	Mat t= Mat(image.size(), CV_8UC1,Scalar(0));
	cout << "print here before trans" << endl;
	t = DeHaze::getInstance()->getTransmission(Icy, A);
	cout << "print here after trans" << endl;
	//获得guide image
	Mat image_src;
	image.copyTo(image_src);
	
	Mat image_gray(image_src.size(), CV_8UC1);
	
	cout << "image_gray rows " << image_gray.rows << " image_gray cols " << image_gray.cols << " image_gray channels " << image_gray.channels() << endl;
	cvtColor(image_src, image_gray, CV_BGR2GRAY);
	cout << "image_gray rows " << image_gray.rows << " image_gray cols " << image_gray.cols << " image_gray channels " << image_gray.channels() << endl;
	imshow("ddd", image_gray);
	Mat guide = DeHaze::getInstance()->getimage(image_gray);
	imshow("guide", guide);
	int r = 8;
	double eps = 0.04;
	int s = 8;
	Mat q = DeHaze::getInstance()->guildFilter(guide, t, r, eps,s);
	imshow("q", q);
	//Mat guidedt;
	//q.copyTo(guidedt);
	//imshow("bbb", image);
	Mat img = Mat(image.size(), CV_8UC3,Scalar(0,0,0));
	imshow("img", img);
	image.copyTo(img);
	Mat deHAZE = Mat(image.size(), CV_8UC3,Scalar(0,0,0));
	deHAZE=DeHaze::getInstance()->getDehazedImage(img, q, A);
	cout << "print here" << endl;
	imshow("原图", image);
	
	imshow("去雾后的图", deHAZE);
	imwrite("D:\\imageal\\dehaze\\datadark.jpg", g);
	imwrite("D:\imageal\dehaze\data\\t.jpg", t);
	imwrite("D:\imageal\dehaze\data\\dehazedImage84.jpg", image1);
	//DeHaze::getInstance()->Deinit(guidedt);
	waitKey(0);
	return 0;
}