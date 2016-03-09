#define _CRT_SECURE_NO_WARNINGS

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2\imgproc\imgproc.hpp>	//required for any image related functions
#include <opencv2\core\core.hpp>      // Mat is defined here. 
#include <opencv2\highgui\highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <cxcore.h>		//for machine learning <3

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Histogram1D.h"
#include "ContentFinder.h"
#include "ColorHistogram.h"
#include <ml\ml.hpp>
#include "ml.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/nonfree/nonfree.hpp>



#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD to 1 for Step 1. 0 for step 2 


//typedef struct featureNode 
//	int typeOfNode;	
//0 for node; 1 for interior node; 2 for leaf;


//};


using namespace cv;
using namespace std;


IplImage* convertImageRGBtoHSV(const IplImage* imageRGB)
{
	float fR, fG, fB;
	float fH, fS, fV;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

	// Create a blank HSV image
	IplImage *imageHSV = cvCreateImage(cvGetSize(imageRGB), 8, 3);
	if (!imageHSV || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
		printf("ERROR in convertImageRGBtoHSV()! Bad input image.\n");
		exit(1);
	}

	int h = imageRGB->height;		// Pixel height.
	int w = imageRGB->width;		// Pixel width.
	int rowSizeRGB = imageRGB->widthStep;	// Size of row in bytes, including extra padding.
	char *imRGB = imageRGB->imageData;	// Pointer to the start of the image pixels.
	int rowSizeHSV = imageHSV->widthStep;	// Size of row in bytes, including extra padding.
	char *imHSV = imageHSV->imageData;	// Pointer to the start of the image pixels.

	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x * 3);
			int bB = *(uchar*)(pRGB + 0);	// Blue component
			int bG = *(uchar*)(pRGB + 1);	// Green component
			int bR = *(uchar*)(pRGB + 2);	// Red component

			// Convert from 8-bit integers to floats.
			fR = bR * BYTE_TO_FLOAT;
			fG = bG * BYTE_TO_FLOAT;
			fB = bB * BYTE_TO_FLOAT;

			// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
			float fDelta;
			float fMin, fMax;
			int iMax;
			// Get the min and max, but use integer comparisons for slight speedup.
			if (bB < bG) {
				if (bB < bR) {
					fMin = fB;
					if (bR > bG) {
						iMax = bR;
						fMax = fR;
					}
					else {
						iMax = bG;
						fMax = fG;
					}
				}
				else {
					fMin = fR;
					fMax = fG;
					iMax = bG;
				}
			}
			else {
				if (bG < bR) {
					fMin = fG;
					if (bB > bR) {
						fMax = fB;
						iMax = bB;
					}
					else {
						fMax = fR;
						iMax = bR;
					}
				}
				else {
					fMin = fR;
					fMax = fB;
					iMax = bB;
				}
			}
			fDelta = fMax - fMin;
			fV = fMax;				// Value (Brightness).
			if (iMax != 0) {			// Make sure it's not pure black.
				fS = fDelta / fMax;		// Saturation.
				float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
				if (iMax == bR) {		// between yellow and magenta.
					fH = (fG - fB) * ANGLE_TO_UNIT;
				}
				else if (iMax == bG) {		// between cyan and yellow.
					fH = (2.0f / 6.0f) + (fB - fR) * ANGLE_TO_UNIT;
				}
				else {				// between magenta and cyan.
					fH = (4.0f / 6.0f) + (fR - fG) * ANGLE_TO_UNIT;
				}
				// Wrap outlier Hues around the circle.
				if (fH < 0.0f)
					fH += 1.0f;
				if (fH >= 1.0f)
					fH -= 1.0f;
			}
			else {
				// color is pure Black.
				fS = 0;
				fH = 0;	// undefined hue
			}

			// Convert from floats to 8-bit integers.
			int bH = (int)(0.5f + fH * 255.0f);
			int bS = (int)(0.5f + fS * 255.0f);
			int bV = (int)(0.5f + fV * 255.0f);

			// Clip the values to make sure it fits within the 8bits.
			if (bH > 255)
				bH = 255;
			if (bH < 0)
				bH = 0;
			if (bS > 255)
				bS = 255;
			if (bS < 0)
				bS = 0;
			if (bV > 255)
				bV = 255;
			if (bV < 0)
				bV = 0;

			// Set the HSV pixel components.
			uchar *pHSV = (uchar*)(imHSV + y*rowSizeHSV + x * 3);
			*(pHSV + 0) = bH;		// H component
			*(pHSV + 1) = bS;		// S component
			*(pHSV + 2) = bV;		// V component
		}
	}
	return imageHSV;
}

IplImage* convertImageHSVtoRGB(const IplImage *imageHSV)
{
	float fH, fS, fV;
	float fR, fG, fB;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

	// Create a blank RGB image
	IplImage *imageRGB = cvCreateImage(cvGetSize(imageHSV), 8, 3);
	if (!imageRGB || imageHSV->depth != 8 || imageHSV->nChannels != 3) {
		printf("ERROR in convertImageHSVtoRGB()! Bad input image.\n");
		exit(1);
	}

	int h = imageHSV->height;			// Pixel height.
	int w = imageHSV->width;			// Pixel width.
	int rowSizeHSV = imageHSV->widthStep;		// Size of row in bytes, including extra padding.
	char *imHSV = imageHSV->imageData;		// Pointer to the start of the image pixels.
	int rowSizeRGB = imageRGB->widthStep;		// Size of row in bytes, including extra padding.
	char *imRGB = imageRGB->imageData;		// Pointer to the start of the image pixels.
	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			// Get the HSV pixel components
			uchar *pHSV = (uchar*)(imHSV + y*rowSizeHSV + x * 3);
			int bH = *(uchar*)(pHSV + 0);	// H component
			int bS = *(uchar*)(pHSV + 1);	// S component
			int bV = *(uchar*)(pHSV + 2);	// V component

			// Convert from 8-bit integers to floats
			fH = (float)bH * BYTE_TO_FLOAT;
			fS = (float)bS * BYTE_TO_FLOAT;
			fV = (float)bV * BYTE_TO_FLOAT;

			// Convert from HSV to RGB, using float ranges 0.0 to 1.0
			int iI;
			float fI, fF, p, q, t;

			if (bS == 0) {
				// achromatic (grey)
				fR = fG = fB = fV;
			}
			else {
				// If Hue == 1.0, then wrap it around the circle to 0.0
				if (fH >= 1.0f)
					fH = 0.0f;

				fH *= 6.0;			// sector 0 to 5
				fI = floor(fH);		// integer part of h (0,1,2,3,4,5 or 6)
				iI = (int)fH;			//		"		"		"		"
				fF = fH - fI;			// factorial part of h (0 to 1)

				p = fV * (1.0f - fS);
				q = fV * (1.0f - fS * fF);
				t = fV * (1.0f - fS * (1.0f - fF));

				switch (iI) {
				case 0:
					fR = fV;
					fG = t;
					fB = p;
					break;
				case 1:
					fR = q;
					fG = fV;
					fB = p;
					break;
				case 2:
					fR = p;
					fG = fV;
					fB = t;
					break;
				case 3:
					fR = p;
					fG = q;
					fB = fV;
					break;
				case 4:
					fR = t;
					fG = p;
					fB = fV;
					break;
				default:		// case 5 (or 6):
					fR = fV;
					fG = p;
					fB = q;
					break;
				}
			}

			// Convert from floats to 8-bit integers
			int bR = (int)(fR * FLOAT_TO_BYTE);
			int bG = (int)(fG * FLOAT_TO_BYTE);
			int bB = (int)(fB * FLOAT_TO_BYTE);

			// Clip the values to make sure it fits within the 8bits.
			if (bR > 255)
				bR = 255;
			if (bR < 0)
				bR = 0;
			if (bG > 255)
				bG = 255;
			if (bG < 0)
				bG = 0;
			if (bB > 255)
				bB = 255;
			if (bB < 0)
				bB = 0;

			// Set the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x * 3);
			*(pRGB + 0) = bB;		// B componentREDS
			*(pRGB + 1) = bG;		// G component
			*(pRGB + 2) = bR;		// R component
		}
	}
	return imageRGB;
}




















int main(){

	FILE *fp;
	fp = fopen("histogram.txt", "a");


	//read an image 
	//assuming that this is from the phone

	IplImage* iplSrc = cvLoadImage("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\A. DATA\\vegetables\\acelgas 1.jpg", 1);
	IplImage* iplGray = cvCreateImage(cvGetSize(iplSrc), IPL_DEPTH_8U, 1);
	CvMemStorage* storage = cvCreateMemStorage(0);
	//convert to grayscalebr
	cvCvtColor(iplSrc, iplGray, CV_BGR2GRAY);





	//-----------
	//create another copy of iplGray 
	IplImage *iplBinary = cvCloneImage(iplGray);
	cvThreshold(iplGray, iplBinary, 80, 255, CV_THRESH_BINARY);

	//-----------
	//convert to HSV 
	IplImage* imHSV;
	IplImage* imRGB;
	IplImage* iplSrcCopy;


	//create another copy of iplSrc
	//convert this copy to HSV
	iplSrcCopy = cvCloneImage(iplSrc);
	IplImage* imgHSV1 = cvCreateImage(cvGetSize(iplSrcCopy), 8, 3);		//creating new blank image
	cvCvtColor(iplSrcCopy, imgHSV1, CV_BGR2HSV);						//storing HSV version of iplSrcCopy to imgHSV1

	//threshold yellow out of the copy of iplSrc
	IplImage* imgThreshed = cvCreateImage(cvGetSize(iplSrcCopy), 8, 1);
	cvInRangeS(iplSrc, Scalar(0, 90, 0), Scalar(140, 255, 255), imgThreshed);	//imgThreshed is thresholded

	//convert imgThreshed to mat
	Mat matImgThreshed = cvarrToMat(imgThreshed);

	//display yellow thresholded 
	//	imshow("this is for yellow", matImgThreshed);

	//create another copy of iplSrc
	imRGB = cvCloneImage(iplSrc);

	//imHSV = cvCreateImage(...);	<---- allocate a new blank image (wrong!)

	//convert to HSV
	imHSV = convertImageRGBtoHSV(imRGB);	//allocates a new HSV image

	//convert imHSV to mat look further down for it :) (a)


	//-----------
	//this is done to prevent a lot of false circles from being detected
	cvSmooth(iplGray, iplGray, CV_GAUSSIAN, 7, 7);

	IplImage* iplCanny = cvCreateImage(cvGetSize(iplSrc), IPL_DEPTH_8U, 1);
	IplImage* rgbcanny = cvCreateImage(cvGetSize(iplSrc), IPL_DEPTH_8U, 3);
	cvCanny(iplGray, iplCanny, 50, 10, 3);



	//(a)
	//convert IplImage file to cv::Mat object	
	Mat matSrc = cvarrToMat(iplSrc);
	Mat matImgHSV = cvarrToMat(imHSV);
	Mat matSrcCopy = cvarrToMat(iplSrcCopy);

	cv::Mat src_gray;
	if (!matSrc.data)
	{
		return -1;
	}


	//START HERE IF YOU WANT TO TAKE CONTOURS OF AN IMAGE
	//(b)convert HSV image to Binary image
	Mat binaryHSV;


	//this works for red cooked chorizo, meat, etc and YELLOW??? tsk
	inRange(matImgHSV, Scalar(0, 90, 0), Scalar(140, 255, 255), binaryHSV);
	//		inRange(matImgHSV, Scalar(20,100,100), Scalar(30, 255, 255), binaryHSV);




	// find contours from binary image
	//The easiest way to look at the two largest contours is to simply look at the contours size

	int i;
	vector<vector<Point>>contours;
	findContours(binaryHSV, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); //find contours
	vector<double> areas(contours.size());

	//gather all the contours and put it in areas
	//find largest contour area
	printf("contour size: %d\n", contours.size());

	for (i = 0; i < contours.size(); i++)
	{
		areas[i] = contourArea(Mat(contours[i]));
	}

	//get index of largest contour
	double max;
	Point maxPosition;
	minMaxLoc(Mat(areas), 0, &max, 0, &maxPosition);

	//draw largest contour
	//doble nah
	drawContours(binaryHSV, contours, maxPosition.y, Scalar(255), CV_FILLED);

	//draw bounding rectangle around largest contour
	Point center;
	Rect r;
	if (contours.size() >= 1)
	{
		r = boundingRect(contours[maxPosition.y]);
		rectangle(binaryHSV, r.tl(), r.br(), CV_RGB(255, 0, 0), 3, 8, 0);	//draw rectangle				
		Mat roi(binaryHSV, r);
		Mat mask1(roi.size(), roi.type(), Scalar::all(0));

		printf("matSrc size: %d\n", matSrc.size());														//TOTAL NUMBER OF PIXELS IN CHOSEN FOOD



		//get the number of pixels of the coin
		//STEP 1: CROP THE COIN
		//get the Rect containing the circle
		//obtain the image ROI:
		//make a black mask, same size:
		//with a white,filled circle in it:
		//combine roi and mask:
		cv::Mat food1_cropped = roi & mask1;

		imshow("food1_cropped", food1_cropped);
		printf("coin size: %d\n", food1_cropped.size());

	}



	//get centroid
	center.x = r.x + (r.width / 2);
	center.y = r.y + (r.height / 2);
	//print x and y coordinates on image
	char x[15], y[6], str1[50];	//text to be written on food;
	_itoa(center.x, x, 10);
	_itoa(center.y, y, 10);



	//std::stringstream ss;
	//ss << 0.5f;
	//std::string s = ss.str();

	//then you can pass s to PutText







	//	putText(matSrc, strcat(x, y), center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);
	//	char str2[] = "Carrots";
	//	putText(matSrc, str2, center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


	//---------------
	//histogram for colored images
	//Separate the images in 3 places (B,G,R)
	vector<Mat> bgr_planes;
	split(matSrc, bgr_planes);
	//establish the number of bins
	int histSize = 256;
	//set the ranges (for B,G,R)
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat b_hist, g_hist, r_hist;

	//compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	//draw the histograms for B,G,R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	//normalizze the result to [0,histImage.rows]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	//display
	namedWindow("calcHist", CV_WINDOW_AUTOSIZE);
	imshow("calcHist", histImage);

	//-----------------

	//---------------


	//convert IplImage file to cv::Mat object	
	Mat matCanny = cvarrToMat(iplCanny);
	Mat matRgbCanny = cvarrToMat(rgbcanny);
	Mat matGray = cvarrToMat(iplGray);
	Mat matBinary = cvarrToMat(iplBinary);


	//------------------------------------------------------------------------
	//HISTOGRAM - is a simple table that gives the number of pixels 
	//that have a given value in an image (or sometime a set of images). 
	//read input image
	cv::Mat graySrcForHistogram;
	cv::Mat histoImage = matSrc;	//open in b&w
	cvtColor(matSrc, graySrcForHistogram, CV_BGR2GRAY);


	//the histogram object
	Histogram1D h;
	//compute the histogram
	cv::MatND histo = h.getHistogram(histoImage);
	//histo object is a 1D array with 256 entries	
	//read each bin by simply looping over this array
	for (int i = 0; i < 256; i++)
	{
		fprintf(fp, "%.0f\t", histo.at<float>(i));
		//		printf("value %d = %.0f\n", i, histo.at<float>(i));
	}

	//resize the original img, as half size of original cols and rows
	//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
	cv::Mat histoimage_resize;
	cv::resize(histoImage, histoimage_resize, cv::Size2i(histoImage.cols / 2, histoImage.rows / 2));

	cv::resize(histoImage, histoimage_resize, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);


	//visualize the original and resized image in window: lena and lena_size respectively.	
	cv::namedWindow("histoImageResizeW");
	cv::imshow("Cropped food", histoimage_resize);

	cv::imwrite("histogram.png", histoimage_resize);


	//****** DRAW ZE HISTOGRAM 
	cv::namedWindow("Histogram");
	cv::imshow("Histogram", h.getHistogramImage(histoimage_resize));
	//****** END OF DRAW ZE HISTOGRAM 

	//thresholding to create a binary image
	//here we threshold the image at the minimum value just before the
	//increase toward the high peak of the histogram (gray value 102)
	cv::Mat thresholded;
	cv::threshold(histoimage_resize, thresholded, 84, 130, cv::THRESH_BINARY);
	cv::namedWindow("binaryImage");
	//cv::imshow("binaryImage", thresholded);


	//end of histogram
	//------------------------------------------------------------------------







	cv::Mat noBackround_image;
	//MIN B:77 G:0 R:30	   //MAX B:130 G:68 R:50
	cv::inRange(matSrc, cv::Scalar(0, 0, 50), cv::Scalar(100, 255, 0), noBackround_image);
	cv::bitwise_not(noBackround_image, noBackround_image);
	cv::imwrite("so_inrange.png", noBackround_image);


	//imshow("no background", noBackround_image);


	//resize images
	cv::Mat reduced_src, matCanny_reduced, matRgbCanny_reduced, matGray_reduced, matBinary_reduced, matDrawing_reduced, nobackground_reduced, matImHSV_reduced;
	cv::Mat binaryHSV_reduced, matImgHSV_reduced;
	cv::resize(noBackround_image, nobackground_reduced, cv::Size2i(noBackround_image.cols / 2, noBackround_image.rows / 2));

	cv::resize(matSrc, reduced_src, cv::Size2i(matSrc.cols / 2, matSrc.rows / 2));

	//	cv::resize(matSrc, reduced_src, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);



	cv::resize(matCanny, matCanny_reduced, cv::Size2i(matCanny.cols / 2, matCanny.rows / 2));
	cv::resize(matRgbCanny, matRgbCanny_reduced, cv::Size2i(matRgbCanny.cols / 2, matRgbCanny.rows / 2));
	cv::resize(matBinary, matBinary_reduced, cv::Size2i(matBinary.cols / 2, matBinary.rows / 2));
	cv::resize(matImgHSV, matImHSV_reduced, cv::Size2i(matImgHSV.cols / 2, matImgHSV.rows / 2));
	cv::resize(binaryHSV, binaryHSV_reduced, cv::Size2i(binaryHSV.cols / 2, binaryHSV.rows / 2));
	//cv::resize(binaryHSV, binaryHSV_reduced, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);

	cv::resize(matImgHSV, matImgHSV_reduced, cv::Size2i(matImgHSV.cols / 2, matImgHSV.rows / 2));
	cv::resize(matGray, matGray_reduced, cv::Size2i(matGray.cols / 2, matGray.rows / 2));














	//------------------------------------------------------------------------------------------------------------
	//save to yml file for training
	FileStorage fs("PositiveCHICKEN.yml", FileStorage::WRITE);

	char FullFileName[100];
	char FirstFileName[100] = "C:\\Users\\Daphne Leah Sabang\\Desktop\\posCHICKEN\\pos-";
	int num_files = 38, num_features;
	int img_area = 1500;

	for (int i = 0; i < num_files; i++)
	{
		sprintf_s(FullFileName, "%s%d.jpg", FirstFileName, i + 1);

		//read image file  
		Mat img_yml, img_yml_gray;
		img_yml = imread(FullFileName);	// I used 0 for grayscale
		Mat training_mat(num_files, img_area, CV_32FC1);

		Mat labels(num_files, 1, CV_32FC1);

		fs << "img" << img_yml;
	}


	fs << "largest contour" << binaryHSV_reduced;



	fs << "frameCount" << 5;
	time_t rawtime; time(&rawtime);
	fs << "calibrationDate" << asctime(localtime(&rawtime));
	Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
	fs << "features" << "[";
	for (int i = 0; i < 3; i++)
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++)
			fs << ((lbp >> j) & 1);
		fs << "]" << "}";
	}
	fs << "]";


	//	return 0;


	//save to yml file for training
	//------------------------------------------------------------------------------------------------------------



































	//-------------
	//find contours version 1
	Mat gray;
	cvtColor(matSrc, gray, CV_BGR2GRAY);
	Canny(gray, gray, 40, 100, 3);

	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	findContours(gray, contours1, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//draw contours
	Mat drawing = Mat::zeros(gray.size(), CV_8UC3);

	printf("contours: %d\n", contours1.size());
	for (int i = 0; i < contours1.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours1, i, color, 2, 8, hierarchy, 0, Point());
	}

	cv::resize(gray, matGray_reduced, cv::Size2i(gray.cols / 2, gray.rows / 2));
	cv::resize(drawing, matDrawing_reduced, cv::Size2i(drawing.cols / 2, drawing.rows / 2));

	//-------------
	//UNCOMMENT IF NEXT DOESN'T WERK
	//find contours version 2
	//	cv::Mat matContourSrc=matSrc;
	//	Mat gray2, gray2_reduced;
	//	cvtColor(matContourSrc, gray2, CV_BGR2GRAY);
	//	threshold(gray2, gray2, 90, 200, THRESH_BINARY_INV); //threshold the gray

	//	cv::resize(gray2, gray2_reduced, cv::Size2i(gray2.cols / 2, gray2.rows / 2));
	//	imshow("gray", gray2_reduced);
	//	int largest_area = 0;
	//	int largest_contour_index = 0;
	//	cv::Mat cropped;

	//--------------
	cv::Mat matContourSrc = matSrc;
	Mat gray2, gray2_reduced;
	cvtColor(matContourSrc, gray2, CV_BGR2GRAY);
	threshold(gray2, gray2, 90, 200, THRESH_BINARY_INV); //threshold the gray

	cv::resize(gray2, gray2_reduced, cv::Size2i(gray2.cols / 2, gray2.rows / 2));
	//imshow("gray", gray2_reduced);

	int largest_area = 0;
	int largest_contour_index = 0;
	cv::Mat cropped;


	//--------------

	Rect bounding_rect;
	vector<vector<Point>> contours2; //vector for storing contour
	vector<Vec4i>hierarchy1;
	findContours(gray2, contours2, hierarchy1, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);



	//iterate through each contour
	for (int i = 0; i < contours2.size(); i++)
	{
		//find the area of contour
		double a = contourArea(contours2[i], false);
		if (a > largest_area){
			largest_area = a;

			printf("area=%.2f\n", a);
			//store the index of largest contour
			largest_contour_index = 1;
			//find the bounding rectangle for biggest contour
			bounding_rect = boundingRect(contours2[i]);

			//cropped = gray2(bounding_rect).clone();


		}
	}

	Scalar color(80, 255, 255); //color the contour in the 
	//draw the contour and rectangle
	//	drawContours(matContourSrc, contours2, largest_contour_index, color, CV_FILLED, 8, hierarchy1);
	rectangle(matContourSrc, r, Scalar(0, 255, 0), 2, 8, 0);


	//cropping off the food image
	cropped = matContourSrc(r).clone();

	//-------------------









	//--------------------

	cv::Mat matGray2_reduced, matContourSrc_reduced, cropped_reduced, foreground_reduced;
	cv::resize(matContourSrc, matContourSrc_reduced, cv::Size2i(matContourSrc.cols / 2, matContourSrc.rows / 2));
	cv::resize(gray2, matGray2_reduced, cv::Size2i(gray2.cols / 2, gray2.rows / 2));
	cv::resize(matContourSrc, matContourSrc_reduced, cv::Size2i(matContourSrc.cols / 2, matContourSrc.rows / 2));
	cv::resize(cropped, cropped_reduced, cv::Size2i(cropped.cols / 2, cropped.rows / 2));
	//	cv::resize(foreground, foreground_reduced, cv::Size2i(foreground.cols / 2, foreground.rows / 2));

	namedWindow("contour version 2", CV_WINDOW_AUTOSIZE);
	//imshow("contour version 2", matContourSrc_reduced);



	//imshow("cropped", cropped_reduced);

	//-------------

	//convert it to gray
	cvtColor(cropped_reduced, src_gray, CV_BGR2GRAY);


	//--------------																						//COME BACK HERE
	//binarize image version 1
	cv::Mat copy_gray = src_gray;
	cv::Mat foodBinaryHSV;
	//	cv::threshold(cropped_reduced, copy_gray, 90.0, 255.0, THRESH_BINARY);
	//	cv::threshold(cropped_reduced, copy_gray, 50.0, 255.0, THRESH_BINARY);

	inRange(copy_gray, Scalar(0, 90, 0), Scalar(50, 255, 255), foodBinaryHSV);

	namedWindow("binary", CV_WINDOW_AUTOSIZE);
	//imshow("binary", copy_gray);															//BLACK AND WHITE VERSION OF CROPPED FOOOD SO THIS IS GOOD



	float coinAccumulate = 0.0;


	//--------------
	//reduce the noise to avoid false circle detection
	GaussianBlur(matGray, matGray, Size(9, 9), 2, 2);
	cv::medianBlur(matGray, matGray, 3);


	vector<Vec3f> circles;
	//apply the Hough transform to find the circles
	HoughCircles(matGray, circles, CV_HOUGH_GRADIENT, 1, matGray.rows / 8, 50, 45, 0, 0);	//60 is a nice value

	//draw the circles detected
	printf("circles: %d\n", circles.size());
	if (circles.size() == 0)
	{
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat foodGraySrcForHistogram; 
		cv::Mat foodHistoImage = copy_gray;	//open food in b&w

		if (foodHistoImage.empty()){
			printf("cannot access frame");
			return -1;
		}

		//		cvtColor(foodHistoImage, foodGraySrcForHistogram, CV_BGR2GRAY);

		imshow("binaryHSV_reduced", binaryHSV_reduced);

		cv::Mat foodGraySrcForHistogram_resize;
		cv::resize(foodGraySrcForHistogram, foodGraySrcForHistogram_resize, cv::Size2i(foodGraySrcForHistogram.cols / 2, foodGraySrcForHistogram.rows / 2));

		imshow("binary image of cropped food", foodGraySrcForHistogram_resize);

		//the histogram object
		Histogram1D foodH;
		//compute the histogram
		cv::MatND foodHisto = foodH.getHistogram(foodHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array



		float foodAccumulate = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", foodHisto.at<float>(i));
			printf("A food value %d = %.0f\n", i, foodHisto.at<float>(i));

			if (i != 0)
			{
				foodAccumulate += foodHisto.at<float>(i);
			}
		}
		printf("coin accumulate: %d\n", coinAccumulate);
		printf("total pixels in food = %.0f\n", foodAccumulate);

		float totalCoinsThatCanFitInFood;																	//count the number of grams in food
		totalCoinsThatCanFitInFood = foodAccumulate / coinAccumulate;
		//if (totalCoinsThatCanFitInFood > 1)
		//{
		//	totalCoinsThatCanFitInFood = 1;
		//}
		printf("number of coins that could fit in food: %.0f\n", totalCoinsThatCanFitInFood);
		float totalGrams = totalCoinsThatCanFitInFood * 3.0;			//2.9 is for vegetables
		printf("this has %.0f grams\n", totalGrams);







		std::stringstream ss;
		ss << foodAccumulate;
		std::string s = ss.str();

		//then you can pass s to PutText




		//resize the original img, as half size of original cols and rows
		//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
		cv::Mat foodHistoimage_resize;
		cv::resize(foodHistoImage, foodHistoimage_resize, cv::Size2i(foodHistoImage.cols / 2, foodHistoImage.rows / 2));

		//visualize the original and resized image in window.	
		cv::namedWindow("coinHhistoImageResizeW");
		cv::imshow("histoImageResizeW", foodHistoimage_resize);

		cv::imwrite("histogram.png", foodHistoimage_resize);

		imshow("HHHHH", foodHistoImage);

		//****** DRAW ZE HISTOGRAM 
		cv::namedWindow("food Histogram");
		cv::imshow("food Histogram", h.getHistogramImage(foodHistoimage_resize));
		//****** END OF DRAW ZE coin HISTOGRAM 



		//SAVE CROPPED FOOD TO A SEPARATE FOLDER
		IplImage* iplSrc = cvLoadImage("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\FINAL DEFENSE DATA\\cropped food items\\pineapple 1.jpg", 1);
		//convert imgThreshed to mat
		Mat matIplSrc = cvarrToMat(iplSrc);

		if (!matIplSrc.data)
		{
			printf(" No image data \n ");
			return -1;
		}

		Mat gray_image;
		cvtColor(matIplSrc, gray_image, CV_BGR2GRAY);

		imwrite("C:\\Users\Daphne Leah Sabang\Documents\School\2ND SEM 2014-2015\THESIS\FINAL DEFENSE DATA\foodROI.jpg", gray_image);

		namedWindow("newImage", CV_WINDOW_AUTOSIZE);
		namedWindow("Gray image", CV_WINDOW_AUTOSIZE);

		imshow("newImage", matIplSrc);
		imshow("Gray image", gray_image);


		//END OF SAVE CROPPED FOOD TO A SEPARATE FOLDER SNIPPET















		//thresholding to create a binary image
		//here we threshold the image at the minimum value just before the
		//increase toward the high peak of the histogram (gray value 102)
		cv::Mat food_thresholded;
		cv::threshold(foodHistoImage, food_thresholded, 84, 130, cv::THRESH_BINARY);
		cv::namedWindow("food binaryImage");
		cv::imshow("food binaryImage", food_thresholded);


		//end of food histogram
		//------------------------------------------------------------------------







		//----------------------------------------------------------------------------------------------------------------------------------------------------------------
		//GET ALL THE REDS in food if there is..

		// Convert input image to HSV


		cv::Mat hsv_image;
		cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the red pixels
		cv::Mat lower_red_hue_range;
		cv::Mat upper_red_hue_range;
		cv::inRange(hsv_image, cv::Scalar(0, 40, 40), cv::Scalar(7, 255, 255), lower_red_hue_range);
		cv::inRange(hsv_image, cv::Scalar(160, 40, 40), cv::Scalar(179, 255, 255), upper_red_hue_range);



		// Combine the above two images
		cv::Mat red_hue_image;
		cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);


		//before blurring it out, get the total number of red pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat redGraySrcForHistogram;
		cv::Mat redHistoImage = red_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);


		//the histogram object
		Histogram1D redH;
		//compute the histogram
		cv::MatND redHisto = redH.getHistogram(redHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonRedAccumulate = 0.0, totalRedPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", redHisto.at<float>(i));
			nonRedAccumulate += redHisto.at<float>(i);
		}
		totalRedPixels = foodAccumulate - nonRedAccumulate;

		//END OF GETTING ALL THE RED
		//--------------------------------------------***************************************


		//--------------------------------------------***************************************
		//GET ALL THE BROWN
		//it can also get YELLOWS
		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the brown pixels
		cv::Mat lower_brown_hue_range;
		cv::Mat upper_brown_hue_range;
		cv::inRange(hsv_image, cv::Scalar(10, 79, 30), cv::Scalar(20, 255, 255), lower_brown_hue_range);				//brown = HUE =10-20; S=79-255;V=30-255
		cv::inRange(hsv_image, cv::Scalar(10, 79, 30), cv::Scalar(20, 255, 255), upper_brown_hue_range);				//brown = HUE =10-20; S=79-255;V=30-255

		// Combine the above two images
		cv::Mat brown_hue_image;
		cv::addWeighted(lower_brown_hue_range, 1.0, upper_brown_hue_range, 1.0, 0.0, brown_hue_image);

		//before blurring it out, get the total number of brown pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat brownGraySrcForHistogram;
		cv::Mat brownHistoImage = brown_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D brownH;
		//compute the histogram
		cv::MatND brownHisto = brownH.getHistogram(brownHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonBrownAccumulate = 0.0, totalBrownPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", brownHisto.at<float>(i));
			nonBrownAccumulate += brownHisto.at<float>(i);
		}

		totalBrownPixels = foodAccumulate - nonBrownAccumulate;

		printf("Brown 1\n");

		//END OF GETTING ALL THE BROWN
		//--------------------------------------------***************************************












		//--------------------------------------------***************************************
		//GET ALL THE GREEN
		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		cv::Mat lower_green_hue_range;
		cv::Mat upper_green_hue_range;
		cv::inRange(hsv_image, cv::Scalar(31, 69, 56), cv::Scalar(75, 255, 230), lower_green_hue_range);				//green=HUE=31-75;S=69-255;V=56-230
		cv::inRange(hsv_image, cv::Scalar(31, 69, 56), cv::Scalar(75, 255, 230), upper_green_hue_range);				//green=HUE=31-75;S=69-255;V=56-230

		// Combine the above two images
		cv::Mat green_hue_image;
		cv::addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);

		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat greenGraySrcForHistogram;
		cv::Mat greenHistoImage = green_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D greenH;
		//compute the histogram
		cv::MatND greenHisto = greenH.getHistogram(greenHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonGreenAccumulate = 0.0, totalGreenPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", greenHisto.at<float>(i));
			nonGreenAccumulate += greenHisto.at<float>(i);
		}
		totalGreenPixels = foodAccumulate - nonGreenAccumulate;
		//END OF GETTING ALL THE GREEN
		//--------------------------------------------***************************************




		//--------------------------------------------***************************************
		//GET ALL THE WHITE
		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		cv::Mat lower_white_hue_range;
		cv::Mat upper_white_hue_range;
		cv::inRange(hsv_image, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), lower_white_hue_range);				//white=HUE=0-179; S=0-16;V=0-255
		cv::inRange(hsv_image, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), upper_white_hue_range);				//white=HUE=0-179; S=0-16;V=0-255

		// Combine the above two images
		cv::Mat white_hue_image;
		cv::addWeighted(lower_white_hue_range, 1.0, upper_white_hue_range, 1.0, 0.0, white_hue_image);

		//before blurring it out, get the total number of white pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat whiteGraySrcForHistogram;
		cv::Mat whiteHistoImage = white_hue_image;	//open food in b&w
		//		cvtColor(whiteHistoImage, whiteGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D whiteH;
		//compute the histogram
		cv::MatND whiteHisto = whiteH.getHistogram(whiteHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonWhiteAccumulate = 0.0, totalWhitePixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", whiteHisto.at<float>(i));
			nonWhiteAccumulate += whiteHisto.at<float>(i);
		}

		totalWhitePixels = foodAccumulate - nonWhiteAccumulate;

		//END OF GETTING ALL THE white
		//--------------------------------------------***************************************






		//DISPLAY TOTAL COLORS HERE
		printf("Total red pixels: %.0f\n", totalRedPixels);
		printf("Total brown pixels: %.0f\n", totalBrownPixels);
		printf("Total green pixels: %.0f\n", totalGreenPixels);
		printf("Total white pixels: %.0f\n", totalWhitePixels);

		float whatColor = totalRedPixels;
		int color = 1;
		//1-red; 2=brown; 3=yellow; 4=orange; 5=green; 6=white; 7=purple


		if (totalBrownPixels > whatColor)
		{
			whatColor = totalBrownPixels;
			color = 2;
			printf("Brown 2\n");
		}
		if (whatColor < totalGreenPixels)
		{
			whatColor = totalGreenPixels;
			color = 5;
		}
		if (whatColor < totalWhitePixels)
		{
			whatColor = totalWhitePixels;
			color = 6;
		}


		int dominantFoodColor = 0;


		//compare the total number of REDS VS BROWNS
		//then get the dominant color

		//RED: D-D-D-DOMINATING
		//if (totalRedPixels > totalBrownPixels)
		//	if (dominantFoodColor == 0)
		//	{
		//		printf("unclassified\n");
		//	}

		if (whatColor == totalRedPixels || color == 1)
		{
			//dominantFoodColor = 1;	//1 is for red
			printf("RED\n");
			//GET ALL THE REDS CUZ YOU ARE SURE THE FOOOD IS SO RED
			//get half of food pixels
			float rednessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float rednessRange2 = foodAccumulate / 2;			//sample: 3000 range: 1500
			float rednessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total red pixels: %.0f\n", totalRedPixels);
			printf("redness range1: %.0f\n", rednessRange1);
			printf("redness range2: %.0f\n", rednessRange2);
			printf("redness range3: %.0f\n", rednessRange3);


			if (totalRedPixels > 900000 && totalRedPixels < 1200000)
			{
				printf("red 1 it reached 1 M red pixels, so offset\n");
				//if total red pixels reaches 1Melyon 
				totalRedPixels -= 200000;								//deduct so it can fall to the last range :)
			}

			if (totalRedPixels < rednessRange2 && totalRedPixels > rednessRange1)
			{
				printf("red 2\n");
				dominantFoodColor = 1;
			}

			if (totalRedPixels > 1000 && totalRedPixels <= 300000)		//5000 because we dont want to catch the browns here too.
			{
				printf("red 3 purple\n");
				dominantFoodColor = 4;
			}


			//check if red pixels is half or more than half the total food pixels

			//check if red pixels is half or more than half the total food pixels

			if (totalRedPixels > rednessRange1 && totalRedPixels < rednessRange2)
			{
				printf("red 4 dark purple\n");
				dominantFoodColor = 7;
				printf("dark purplish");
			}

			if (totalRedPixels < rednessRange1)
			{
				printf("red 5 orangee\n");
				dominantFoodColor = 4;

			}

			if (totalRedPixels > 1000 && totalRedPixels <= 300000)		//5000 because we dont want to catch the browns here too.
			{
				printf("red 6 orangee\n");
				dominantFoodColor = 4;
			}


			else if (totalRedPixels < rednessRange3 && totalRedPixels > rednessRange2)
			{
				printf("red 7\n");
				dominantFoodColor = 1;

			}
			else if (totalRedPixels > rednessRange3 && totalRedPixels < rednessRange3 + 100)
			{
				printf("red 8\n");
				dominantFoodColor = 1;
				//check for contours here that belong under the red category
			}

			else if (totalRedPixels > rednessRange3)
			{
				printf("red 9\n");
				dominantFoodColor = 1;
				//check for contours here that belong under the red category
			}







			//-----------------


			if (dominantFoodColor == 1)		//RED RED RED
			{
				if (totalRedPixels > rednessRange3 + 50)
				{
					printf("red 10 purple\n");
					dominantFoodColor = 7;	//purple					
				}

				else
				{
					printf("red 11\n");
					printf("soooooo Red\n");
					char str3[] = "fruit";


					char calorieCount[50];
					float oneExchange = 140.0;
					float carbs = 10.0;
					float kcal = 40.0;					//1 exchange = 10 grams carbs = 40kcal
					//let's say totalgrams = 112;		//generated by the system
					if (totalGrams > oneExchange)
					{
						totalGrams /= oneExchange;
						carbs = carbs * totalGrams;
						kcal = carbs * totalGrams;
					}
					//				sprintf_s(calorieCount, "%s has %.0f grams.\n%.0f grams of Carbs = %.0f kcal\n", str3, totalGrams, carbs, kcal);


					cv::Point textOrg(bounding_rect.x, bounding_rect.y);
					putText(matSrc, str3, textOrg, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					cv::Point textOrg1(bounding_rect.x, bounding_rect.y + 30);
					putText(matSrc, "total grams:" + strTotalGrams + "grams", textOrg1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();

					cv::Point textOrg2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "carbohydrates" + strCarbs + "calories", textOrg2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					printf("it can be one of the following: cherry\nbeef tenderloin\nbeefshank\nbreastmeat pitso\ncarabeefshank\nleanpork\napple");
				}

			}















			if (dominantFoodColor == 7)
			{
				printf("soooooo purple\n");
				char str3[] = "fruit";


				char calorieCount[50];
				float oneExchange = 140.0;
				float carbs = 10.0;
				float kcal = 40.0;					//1 exchange = 10 grams carbs = 40kcal
				//let's say totalgrams = 112;		//generated by the system
				if (totalGrams > oneExchange)
				{
					totalGrams /= oneExchange;
					carbs = carbs * totalGrams;
					kcal = carbs * totalGrams;
				}
				//				sprintf_s(calorieCount, "%s has %.0f grams.\n%.0f grams of Carbs = %.0f kcal\n", str3, totalGrams, carbs, kcal);


				cv::Point textOrg(bounding_rect.x, bounding_rect.y);
				putText(matSrc, str3, textOrg, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strT;
				strT << totalGrams;
				std::string strTotalGrams = strT.str();
				cv::Point textOrg1(bounding_rect.x, bounding_rect.y + 30);
				putText(matSrc, "total grams:" + strTotalGrams + "grams", textOrg1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strC;
				strC << carbs;
				std::string strCarbs = strC.str();

				cv::Point textOrg2(bounding_rect.x, bounding_rect.y + 60);
				putText(matSrc, "carbohydrates" + strCarbs + "calories", textOrg2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				printf("can be purplish red\n");

			}












			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat redHistoimage_resize;
			cv::resize(redHistoImage, redHistoimage_resize, cv::Size2i(redHistoImage.cols / 2, redHistoImage.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("redHhistoImageResizeW");
			cv::imshow("redhistoImageResizeW", redHistoimage_resize);

			cv::imwrite("redhistogram.png", redHistoimage_resize);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("red Histogram");
			cv::imshow("red Histogram", h.getHistogramImage(redHistoimage_resize));
			//****** END OF DRAW ZE coin HISTOGRAM 




			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat red_thresholded;
			cv::threshold(redHistoimage_resize, red_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("red binaryImage");
			cv::imshow("red binaryImage", red_thresholded);

			//end of getting the RED histogram
			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

			//get histogram and total number of pixels of black and white of cropped food
			//it is considered red if total pixels of "red"()




			//resize and display
			cv::Mat lower_red_hue_range_reduced, upper_red_hue_range_reduced, red_hue_image_reduced;
			cv::resize(lower_red_hue_range, lower_red_hue_range_reduced, cv::Size2i(lower_red_hue_range.cols / 2, lower_red_hue_range.rows / 2));
			imshow("lower_red_hue_range_reduced", lower_red_hue_range_reduced);

			cv::resize(upper_red_hue_range, upper_red_hue_range_reduced, cv::Size2i(upper_red_hue_range.cols / 2, upper_red_hue_range.rows / 2));
			imshow("upper_red_hue_range_reduced", upper_red_hue_range_reduced);


			cv::resize(red_hue_image, red_hue_image_reduced, cv::Size2i(red_hue_image.cols / 2, red_hue_image.rows / 2));
			imshow("red_hue_image_reduced", red_hue_image_reduced);


			//END OF GETTING ALL THE REDS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------

		}


		//compare the total number of REDS VS BROWNS vs GREENS
		//then get the dominant color

		//BROWN: D-D-D-DOMINATING
		//else if (totalBrownPixels > totalRedPixels)
		else if (whatColor == totalBrownPixels)
		{
			//GET ALL THE BROWNS CUZ FOOD IS SURE TO BE BROWWNN

			printf("\nBROWN\n");
			//GET ALL THE BROWNS CUZ YOU ARE SURE THE FOOOD IS SO BROWN
			//get half of food pixels
			float brownnessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float brownnessRange2 = (foodAccumulate / 2) + 1000;			//sample: 3000 range: 1500
			float brownnessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total brown pixels: %.0f\n", totalBrownPixels);
			printf("brownness range1: %.0f\n", brownnessRange1);
			printf("brownness range2: %.0f\n", brownnessRange2);
			printf("brownness range3: %.0f\n", brownnessRange3);


			printf("Brown 3\n");

			float oneThirdThebrownness = foodAccumulate / 1.5;		//it can be yellow			
			float halfTheBrownness = foodAccumulate / 2.0;



			//--													//below means not within brown range2 and brown range3
			if (totalRedPixels > 1000 && totalRedPixels <= 200000 && !(totalBrownPixels > brownnessRange2 && totalBrownPixels < brownnessRange3))		//5000 because we dont want to catch the browns here too.
			{
				dominantFoodColor = 4;		//4 is for orange :)	

			}

			else if (totalBrownPixels >= brownnessRange2 && totalBrownPixels <= 300000)	//because 300k is yellow already
			{
				dominantFoodColor = 3;
				printf("Brown yellow 1\n");
			}

			else if (totalBrownPixels < brownnessRange2 && totalBrownPixels > brownnessRange1)	//singkamas brown
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 4\n");
			}
			else if (totalBrownPixels > brownnessRange2 && totalBrownPixels <= brownnessRange3)
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 5\n");
			}

			else if (totalBrownPixels <= halfTheBrownness)
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 6\n");
			}

			else if (totalBrownPixels < brownnessRange1)
			{
				dominantFoodColor = 3;	//yellow
				printf("Brown yellow 2\n");
			}

			//--




			else if (dominantFoodColor == 4)
			{
				char str3[] = "brownish Orange";
				putText(matSrc, str3, center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				printf("Brown orange 1");

			}

			else if (dominantFoodColor == 2)
			{
				color = 2;
				printf("Brown 7\n");
				char str3[] = "Bread?";
				//			putText(matSrc, str3, center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				int halfOfBrownRange2And3 = brownnessRange2 + ((brownnessRange3 - brownnessRange2) * 0.5);
				//get the half of range 2 and range 3

				if (totalBrownPixels < brownnessRange2 && totalBrownPixels > brownnessRange1)
				{
					printf("Brown 8\n"); // singkamas brown
				}

				else if (totalBrownPixels < halfOfBrownRange2And3)
				{

					printf("brownish yellow 3\n");


					char calorieCount[50];
					float oneExchange = 140.0;
					float carbs = 10.0;
					float kcal = 40.0;					//1 exchange = 10 grams carbs = 40kcal
					//let's say totalgrams = 112;		//generated by the system
					if (totalGrams > oneExchange)
					{
						totalGrams /= oneExchange;
						carbs = carbs * totalGrams;
						kcal = carbs * totalGrams;
					}
					//				sprintf_s(calorieCount, "%s has %.0f grams.\n%.0f grams of Carbs = %.0f kcal\n", str3, totalGrams, carbs, kcal);


					cv::Point textOrg(bounding_rect.x, bounding_rect.y);
					//				putText(matSrc, str3, textOrg, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					cv::Point textOrg1(bounding_rect.x, bounding_rect.y + 30);
					putText(matSrc, "total grams:" + strTotalGrams + "grams", textOrg1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();

					cv::Point textOrg2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "carbohydrates" + strCarbs + "calories", textOrg2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					printf("it can be one of the following: everything brownish yellow");
					//check for contours here that belong under the red category
				}

				if (totalBrownPixels > halfOfBrownRange2And3)
				{
					printf("Brown 9\n");
					color = 2;
				}


			}
			else if (dominantFoodColor == 3)
			{
				char str3[] = "Mango";

				//char str3[] = "fruit";

				int halfOfBrownRange2And3 = brownnessRange2 + ((brownnessRange3 - brownnessRange2) * 0.5);
				//get the half of range 2 and range 3

				if (totalBrownPixels > halfOfBrownRange2And3)
				{
					printf("Brown 10\n");
				}

				else if (totalBrownPixels < halfOfBrownRange2And3)
				{
					printf("brownish yellow 4\n");


					char calorieCount[50];
					float oneExchange = 140.0;
					float carbs = 10.0;
					float kcal = 40.0;					//1 exchange = 10 grams carbs = 40kcal
					//let's say totalgrams = 112;		//generated by the system
					if (totalGrams > oneExchange)
					{
						totalGrams /= oneExchange;
						carbs = carbs * totalGrams;
						kcal = carbs * totalGrams;
					}
					//				sprintf_s(calorieCount, "%s has %.0f grams.\n%.0f grams of Carbs = %.0f kcal\n", str3, totalGrams, carbs, kcal);


					cv::Point textOrg(bounding_rect.x, bounding_rect.y);
					putText(matSrc, str3, textOrg, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					cv::Point textOrg1(bounding_rect.x, bounding_rect.y + 30);
					putText(matSrc, "total grams:\n" + strTotalGrams, textOrg1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();

					cv::Point textOrg2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "carbohydrates: " + strCarbs + "calories", textOrg2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					printf("this food is so yellow\n");
					printf("it can be one of the following: banana and everything yellow");
					//check for contours here that belong under the red category
				}

			}


			//--





			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat brownHistoimage_resize;
			cv::resize(brownHistoImage, brownHistoimage_resize, cv::Size2i(brownHistoImage.cols / 2, brownHistoImage.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("brownHhistoImageResizeW");
			cv::imshow("brownHhistoImageResizeW", brownHistoimage_resize);

			cv::imwrite("brownhistogram.png", brownHistoimage_resize);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("brown Histogram");
			cv::imshow("brown Histogram", h.getHistogramImage(brownHistoimage_resize));
			//****** END OF DRAW ZE coin HISTOGRAM 



			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat brown_thresholded;
			cv::threshold(brownHistoimage_resize, brown_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("brown binaryImage");
			cv::imshow("brown binaryImage", brown_thresholded);

			//end of getting the RED histogram
			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(brown_hue_image, brown_hue_image, cv::Size(9, 9), 2, 2);


			//get histogram and total number of pixels of black and white of cropped food	
			//resize and display
			cv::Mat lower_brown_hue_range_reduced, upper_brown_hue_range_reduced, brown_hue_image_reduced;
			cv::resize(lower_brown_hue_range, lower_brown_hue_range_reduced, cv::Size2i(lower_brown_hue_range.cols / 2, lower_brown_hue_range.rows / 2));
			imshow("lower_brown_hue_range_reduced", lower_brown_hue_range_reduced);

			cv::resize(upper_brown_hue_range, upper_brown_hue_range_reduced, cv::Size2i(upper_brown_hue_range.cols / 2, upper_brown_hue_range.rows / 2));
			imshow("upper_brown_hue_range_reduced", upper_brown_hue_range_reduced);


			cv::resize(brown_hue_image, brown_hue_image_reduced, cv::Size2i(brown_hue_image.cols / 2, brown_hue_image.rows / 2));
			imshow("brown_hue_image_reduced", brown_hue_image_reduced);

			//END OF GETTING ALL THE BROWNS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------

		}






		//GREEN: D-D-D-DOMINATING
		if (whatColor == totalGreenPixels)
		{
			dominantFoodColor = 5;
			printf("GREEN\n");
			//GET ALL THE GREENS CUZ YOU ARE SURE THE FOOOD IS SO RED
			//get half of food pixels
			float greennessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float greennessRange2 = foodAccumulate / 2;			//sample: 3000 range: 1500
			float greennessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total green pixels: %.0f\n", totalGreenPixels);
			printf("Greenness range1: %.0f\n", greennessRange1);
			printf("Greenness range2: %.0f\n", greennessRange2);
			printf("Greenness range3: %.0f\n", greennessRange3);

			if (totalGreenPixels > 900000 && totalGreenPixels < 1200000)
			{

				//if total red pixels reaches 1Melyon 
				totalGreenPixels -= 200000;								//deduct so it can fall to the last range :)
			}
			if (totalGreenPixels > 1000 && totalGreenPixels <= 300000)		//5000 because we dont want to catch the browns here too.
			{
				//		putText(matSrc, strcat(x, y), center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);
				//				char str2[] = "Vegetables!";
				//				putText(matSrc, str2, center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				printf("this is a vegetable A\n");
				dominantFoodColor = 5;		//4 is for orange :)
			}


			//check if red pixels is half or more than half the total food pixels
			else if (totalGreenPixels > 300000 && totalGreenPixels <= greennessRange3)
			{
				printf("this is a vegetable B\n");
				dominantFoodColor = 5;
				//check for contours here that belong under the red category
			}
			//check if red pixels is half or more than half the total food pixels
			else if (totalGreenPixels <= greennessRange3 || totalGreenPixels < greennessRange3)
			{
				printf("this is a vegetable\n");
				dominantFoodColor = 5;
				//check for contours here that belong under the green category
			}






			//std::stringstream ss;
			//ss << redAccumulate;
			//std::string s = ss.str();

			//then you can pass s to PutText




			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat binaryHSV_reduced;
			cv::resize(binaryHSV_reduced, binaryHSV_reduced, cv::Size2i(binaryHSV_reduced.cols / 2, binaryHSV_reduced.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("greenHhistoImageResizeW");
			cv::imshow("greenhistoImageResizeW", binaryHSV_reduced);

			cv::imwrite("greenhistogram.png", binaryHSV_reduced);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("green Histogram");
			cv::imshow("green Histogram", h.getHistogramImage(binaryHSV_reduced));
			//****** END OF DRAW ZE coin HISTOGRAM 




			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat green_thresholded;
			cv::threshold(binaryHSV, green_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("green binaryImage");
			cv::imshow("green binaryImage", green_thresholded);

			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(green_hue_image, green_hue_image, cv::Size(9, 9), 2, 2);

			//get histogram and total number of pixels of black and white of cropped food
			//it is considered green if total pixels of "green"()




			//resize and display
			cv::Mat lower_green_hue_range_reduced, upper_green_hue_range_reduced, green_hue_image_reduced;
			cv::resize(lower_green_hue_range, lower_green_hue_range_reduced, cv::Size2i(lower_green_hue_range.cols / 2, lower_green_hue_range.rows / 2));
			//imshow("lower_green_hue_range_reduced", lower_green_hue_range_reduced);

			cv::resize(upper_green_hue_range, upper_green_hue_range_reduced, cv::Size2i(upper_green_hue_range.cols / 2, upper_green_hue_range.rows / 2));
			//imshow("upper_green_hue_range_reduced", upper_green_hue_range_reduced);


			cv::resize(green_hue_image, green_hue_image_reduced, cv::Size2i(green_hue_image.cols / 2, green_hue_image.rows / 2));
			//imshow("green_hue_image_reduced", green_hue_image_reduced);


			//END OF GETTING ALL THE GREENS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------

		}
		printf("circles.size()");
	} //end of if (circles.size() == 0)

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);


		//show the cropped coin here if you laik

		//numOfCoinsInFood = food.size() / coinpixels.size() 
		//foodgrams = numOfCoinsInFood * 6.1grams

		//cv::Mat numOfCoinsInFood = ((int)(matSrc.size())/ ((int)(coin_cropped.size());		//CONVERT THIS TO INT SO WE CAN COMPUTE

		//draw the circle center
		circle(matSrc, center, 3, Scalar(50, 50, 50), -1, 8, 0);									//get the circle ing matgray then draw it on matsrc

		//draw the circle outline
		circle(matSrc, center, radius, Scalar(50, 50, 50), 3, 8, 0);



		//get the number of pixels of the coin
		//STEP 1: CROP THE COIN
		//get the Rect containing the circle
		Rect rectCircle(center.x - radius, center.y - radius, radius * 2, radius * 2);
		//obtain the image ROI:
		Mat roi(matGray, rectCircle);
		int numberOfPixelsInMask = cv::countNonZero(matGray);

		//make a bmlack mask, same size:
		Mat mask(roi.size(), roi.type(), Scalar::all(0));
		//with a white,filled circle in it:
		circle(mask, Point(radius, radius), radius, Scalar::all(255), -1);
		//combine roi and mask:m
		cv::Mat coin_cropped = roi & mask;

		imshow("coin cropped", coin_cropped);
		printf("coin size: %d\n", coin_cropped.size());
		printf("coin size v2: %d\n", numberOfPixelsInMask);



		//------------------------------------------------------------------------
		//HISTOGRAM of coin!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat coinGraySrcForHistogram;
		cv::Mat coinHistoImage = coin_cropped;	//at this point, B&W coin has already been cropped here with BLACK background
		//cvtColor(coinHistoImage, coinGraySrcForHistogram, CV_BGR2GRAY);			no need cuz coin_cropped is already BW

		imshow("coinhistoimage", coinHistoImage);

		//the histogram object
		Histogram1D coinH;
		//compute the histogram
		cv::MatND coinHisto = coinH.getHistogram(coinHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", coinHisto.at<float>(i));
			//			printf("coin value %d = %.0f\n", i, coinHisto.at<float>(i));
			if (i != 0)
			{
				coinAccumulate += coinHisto.at<float>(i);
			}
		}
		printf("total pixels in coin = %.0f\n", coinAccumulate);
		//resize the original img, as half size of original cols and rows
		//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
		cv::Mat coinHistoimage_resize;
		cv::resize(coinHistoImage, coinHistoimage_resize, cv::Size2i(coinHistoImage.cols / 2, coinHistoImage.rows / 2));

		//visualize the original and resized image in window: lena and lena_size respectively.	
		cv::namedWindow("coinHhistoImageResizeW");
		cv::imshow("histoImageResizeW", coinHistoimage_resize);

		cv::imwrite("histogram.png", coinHistoimage_resize);


		//****** DRAW ZE HISTOGRAM 
		cv::namedWindow("coin Histogram");
		cv::imshow("coin Histogram", h.getHistogramImage(coinHistoimage_resize));
		//****** END OF DRAW ZE coin HISTOGRAM 

		cv::namedWindow("matGrrr");
		cv::imshow("matGrrr", matGray);


		//thresholding to create a binary image
		//here we threshold the image at the minimum value just before the
		//increase toward the high peak of the histogram (gray value 102)
		cv::Mat coin_thresholded;
		cv::threshold(histoimage_resize, coin_thresholded, 84, 130, cv::THRESH_BINARY);
		cv::namedWindow("coin binaryImage");
		cv::imshow("coin binaryImage", coin_thresholded);


		//end of coin histogram
		//------------------------------------------------------------------------









		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat foodGraySrcForHistogram;
		cv::Mat foodHistoImage = cropped;	//open food in b&w
		cvtColor(foodHistoImage, foodGraySrcForHistogram, CV_BGR2GRAY);




		imshow("binary image of cropped food", foodGraySrcForHistogram);


		cv::Mat foodGraySrcForHistogram_resize;

		//the histogram object
		Histogram1D foodH;
		//compute the histogram
		cv::MatND foodHisto = foodH.getHistogram(foodGraySrcForHistogram);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float foodAccumulate = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", foodHisto.at<float>(i));
			printf("food value %d = %.0f\n", i, foodHisto.at<float>(i));

			if (i != 0)
			{
				foodAccumulate += foodHisto.at<float>(i);
			}
		}
		printf("total pixels in food = %.0f\n", foodAccumulate);

		float totalCoinsThatCanFitInFood;																	//count the number of grams in food
		totalCoinsThatCanFitInFood = foodAccumulate / coinAccumulate;

		cv::resize(foodGraySrcForHistogram, foodGraySrcForHistogram_resize, cv::Size2i(foodGraySrcForHistogram.cols / 2, foodGraySrcForHistogram.rows / 2));

		//if (totalCoinsThatCanFitInFood > 1)
		//{
		//	totalCoinsThatCanFitInFood = 1;
		//}


		printf("number of coins that could fit in food: %.0f\n", totalCoinsThatCanFitInFood);
		float totalGrams = totalCoinsThatCanFitInFood * 2.9;			//2.9 is for vegetables
		printf("this has %.0f grams\n", totalGrams);



		std::stringstream ss;
		ss << foodAccumulate;
		std::string s = ss.str();

		//then you can pass s to PutText




		//resize the original img, as half size of original cols and rows
		//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
		cv::Mat foodHistoimage_resize;
		cv::resize(foodHistoImage, foodHistoimage_resize, cv::Size2i(foodHistoImage.cols / 2, foodHistoImage.rows / 2));

		//visualize the original and resized image in window: lena and lena_size respectively.	
		cv::namedWindow("coinHhistoImageResizeW");
		cv::imshow("histoImageResizeW", foodHistoimage_resize);

		cv::imwrite("histogram.png", foodHistoimage_resize);


		//****** DRAW ZE HISTOGRAM 
		cv::namedWindow("food Histogram");
		cv::imshow("food Histogram", h.getHistogramImage(foodHistoimage_resize));
		//****** END OF DRAW ZE coin HISTOGRAM 




		//thresholding to create a binary image
		//here we threshold the image at the minimum value just before the
		//increase toward the high peak of the histogram (gray value 102)
		cv::Mat food_thresholded;
		cv::threshold(foodHistoImage, food_thresholded, 84, 130, cv::THRESH_BINARY);
		cv::namedWindow("food binaryImage");
		cv::imshow("food binaryImage", food_thresholded);


		//end of food histogram
		//------------------------------------------------------------------------





























		//SAVE CROPPED FOOD TO A SEPARATE FOLDER
		//STEP 1 IN FOOD IMAGE RECOGNITION: SAVING THE CROPPED FOOD TO A FOLDER
		//		IplImage* iplSrc = cvLoadImage("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\FINAL DEFENSE DATA\\cropped food items\\pineapple 1.jpg", 1);
		//convert imgThreshed to mat
		//		Mat matIplSrc = cvarrToMat(iplSrc);
		Mat matCroppedForSaving = cropped;

		if (!matCroppedForSaving.data)
		{
			printf(" No image data \n ");
			//			return -1;
		}

		Mat gray_image;
		cvtColor(matCroppedForSaving, gray_image, CV_BGR2GRAY);

		imwrite("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\FINAL DEFENSE DATA\\cropped food items\\foodROI.jpg", cropped);
		//END OF SAVE CROPPED FOOD TO A SEPARATE FOLDER SNIPPET



		//STEP 2: LOAD SAVED CROPPED IMAGE
		IplImage* iplSrcCroppedFoodROI = cvLoadImage("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\FINAL DEFENSE DATA\\cropped food items\\foodROI.jpg", 1);
		//convert imgThreshed to mat
		//		Mat matIplSrc = cvarrToMat(iplSrc);
		Mat foodCroppedForComparison = cvarrToMat(iplSrcCroppedFoodROI);
		Mat roi_part_of_food = foodCroppedForComparison(Rect(150, 150, 150, 250));
		if (!foodCroppedForComparison.data)
		{
			printf(" No image data \n ");
			//			return -1;
		}

		else{

			cv::namedWindow("Example1");
			imshow("Example1", roi_part_of_food);


			cv::namedWindow("Example 1 histogram");
			cv::imshow("Example 1 histogram", h.getHistogramImage(roi_part_of_food));


			cv::namedWindow("food Cropped for Comparison");
			cv::imshow("food Cropped for Comparison", foodCroppedForComparison);






			//trainData
			FILE *fp;
			char filename[30] = "trainingDataFile.txt";
			fp = fopen(filename, "a+");

			//open training data
			//		IplImage* iplSrcDataTrain = cvLoadImage("C:\\Users\\Daphne Leah Sabang\\Documents\\School\\2ND SEM 2014-2015\\THESIS\\A. DATA\\vegetables\\sikwa 1.jpg", 1);
			//		Mat matSrcDataTrain = cvarrToMat(iplSrcDataTrain);



			//extract 
			//namedWindow("Training Data", CV_WINDOW_AUTOSIZE);
			//imshow("Training Data", matSrcDataTrain);



			//crop roi of training data 
			Mat imgTrain;
			Rect rectTrain;
			/* Get the img from webcam or from file. Get points from mousecallback
			* or define the points
			*/
			rectTrain = Rect(800, 400, 100, 100);
			Mat roiImgTrain, roiImgGrayTrain;
			roiImgTrain = foodCroppedForComparison(rectTrain); /* sliced image */

			namedWindow("Training Data Cropped", CV_WINDOW_AUTOSIZE);
			imshow("Training Data Cropped", roiImgTrain);




			//transform to gray
			cvtColor(roiImgTrain, roiImgGrayTrain, CV_BGR2GRAY);



			//get histogram of roi
			//	Mat gray = imread("image.jpg", 0);
			namedWindow("Train Gray", 1);    imshow("Traing Gray", roiImgGrayTrain);

			// Initialize parametersfood 
			int histSize = 30;    // bin size
			float range[] = { 0, 255 };
			const float *ranges[] = { range };

			// Calculate histogram
			MatND hist;
			calcHist(&roiImgGrayTrain, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

			// Show the calculated histogram in command window
			double total;
			total = roiImgGrayTrain.rows * roiImgGrayTrain.cols;
			for (int h = 0; h < histSize; h++)
			{
				float binVal = hist.at<float>(h);
				cout << " " << binVal;

				fprintf(fp, "%.0f ", binVal);
			}
			fprintf(fp, "\n");

			// Plot the histogram
			int hist_w = 512; int hist_h = 400;
			int bin_w = cvRound((double)hist_w / histSize);

			Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
			normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

			for (int i = 1; i < histSize; i++)
			{
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
			}

			namedWindow("roiImgGrayTrain Result", 1);    imshow("roiImgGrayTrain Result", histImage);



			fclose(fp);


			//end of trainData
		} //end of else part of if (!foodCroppedForComparison.data)





































































		//----------------------------------------------------------------------------------------------------------------------------------------------------------------
		//GET ALL THE REDS in food if there is..

		// Convert input image to HSV


		cv::Mat hsv_image;
		cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the red pixels
		cv::Mat lower_red_hue_range;
		cv::Mat upper_red_hue_range;
		cv::inRange(hsv_image, cv::Scalar(0, 40, 40), cv::Scalar(7, 255, 255), lower_red_hue_range);
		cv::inRange(hsv_image, cv::Scalar(160, 40, 40), cv::Scalar(179, 255, 255), upper_red_hue_range);



		// Combine the above two images
		cv::Mat red_hue_image;
		cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);


		//before blurring it out, get the total number of red pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat redGraySrcForHistogram;
		cv::Mat redHistoImage = red_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);


		//the histogram object
		Histogram1D redH;
		//compute the histogram
		cv::MatND redHisto = redH.getHistogram(redHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonRedAccumulate = 0.0, totalRedPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", redHisto.at<float>(i));
			nonRedAccumulate += redHisto.at<float>(i);
		}

		totalRedPixels = foodAccumulate - nonRedAccumulate;
		//END OF GETTING ALL THE RED
		//--------------------------------------------***************************************


		//--------------------------------------------***************************************
		//GET ALL THE BROWN
		//it can also get YELLOWS
		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the brown pixels
		cv::Mat lower_brown_hue_range;
		cv::Mat upper_brown_hue_range;
		cv::inRange(hsv_image, cv::Scalar(10, 79, 30), cv::Scalar(20, 255, 255), lower_brown_hue_range);				//brown = HUE =10-20; S=79-255;V=30-255
		cv::inRange(hsv_image, cv::Scalar(10, 79, 30), cv::Scalar(20, 255, 255), upper_brown_hue_range);				//brown = HUE =10-20; S=79-255;V=30-255

		// Combine the above two images
		cv::Mat brown_hue_image;
		cv::addWeighted(lower_brown_hue_range, 1.0, upper_brown_hue_range, 1.0, 0.0, brown_hue_image);

		//before blurring it out, get the total number of brown pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat brownGraySrcForHistogram;
		cv::Mat brownHistoImage = brown_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D brownH;
		//compute the histogram
		cv::MatND brownHisto = brownH.getHistogram(brownHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonBrownAccumulate = 0.0, totalBrownPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", brownHisto.at<float>(i));
			nonBrownAccumulate += brownHisto.at<float>(i);
		}
		totalBrownPixels = foodAccumulate - nonBrownAccumulate;

		//END OF GETTING ALL THE BROWN
		//--------------------------------------------***************************************












		//--------------------------------------------***************************************
		//GET ALL THE GREEN
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------
		//GET ALL THE GREEN in food if there is..

		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the green pixels
		cv::Mat lower_green_hue_range;
		cv::Mat upper_green_hue_range;
		cv::inRange(hsv_image, cv::Scalar(31, 69, 56), cv::Scalar(75, 255, 230), lower_green_hue_range);				//green=HUE=31-75;S=69-255;V=56-230
		cv::inRange(hsv_image, cv::Scalar(31, 69, 56), cv::Scalar(75, 255, 230), upper_green_hue_range);				//green=HUE=31-75;S=69-255;V=56-230

		// Combine the above two images
		cv::Mat green_hue_image;
		cv::addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);

		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat greenGraySrcForHistogram;
		cv::Mat greenHistoImage = green_hue_image;	//open food in b&w
		//		cvtColor(redHistoImage, redGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D greenH;
		//compute the histogram
		cv::MatND greenHisto = greenH.getHistogram(greenHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonGreenAccumulate = 0.0, totalGreenPixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", greenHisto.at<float>(i));
			nonGreenAccumulate += greenHisto.at<float>(i);
		}
		totalGreenPixels = foodAccumulate - nonGreenAccumulate;

		//END OF GETTING ALL THE GREEN
		//--------------------------------------------***************************************




		//--------------------------------------------***************************************
		//GET ALL THE WHITE
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------
		//GET ALL THE WHITE in food if there is..

		// Convert input image to HSV
		//			cv::Mat hsv_image;
		//			cv::cvtColor(cropped, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the white pixels
		cv::Mat lower_white_hue_range;
		cv::Mat upper_white_hue_range;
		cv::inRange(hsv_image, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), lower_white_hue_range);				//white=HUE=0-179; S=0-16;V=0-255
		cv::inRange(hsv_image, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), upper_white_hue_range);				//white=HUE=0-179; S=0-16;V=0-255

		// Combine the above two images
		cv::Mat white_hue_image;
		cv::addWeighted(lower_white_hue_range, 1.0, upper_white_hue_range, 1.0, 0.0, white_hue_image);

		//before blurring it out, get the total number of white pixels
		//------------------------------------------------------------------------
		//HISTOGRAM of food!
		//that have a given value in an image (or sometime a set of images). 
		//read input image
		cv::Mat whiteGraySrcForHistogram;
		cv::Mat whiteHistoImage = white_hue_image;	//open food in b&w
		//		cvtColor(whiteHistoImage, whiteGraySrcForHistogram, CV_BGR2GRAY);

		//the histogram object
		Histogram1D whiteH;
		//compute the histogram
		cv::MatND whiteHisto = whiteH.getHistogram(whiteHistoImage);
		//histo object is a 1D array with 256 entries	
		//read each bin by simply looping over this array

		float nonWhiteAccumulate = 0.0, totalWhitePixels = 0.0;
		for (int i = 0; i < 256; i++)
		{
			fprintf(fp, "%.0f\t", whiteHisto.at<float>(i));
			nonWhiteAccumulate += whiteHisto.at<float>(i);
		}
		totalWhitePixels = foodAccumulate - nonWhiteAccumulate;

		//END OF GETTING ALL THE white
		//--------------------------------------------***************************************











		//DISPLAY TOTAL COLORS HERE
		printf("Total red pixels: %.0f\n", totalRedPixels);
		printf("Total brown pixels: %.0f\n", totalBrownPixels);
		printf("Total green pixels: %.0f\n", totalGreenPixels);
		printf("Total white pixels: %.0f\n", totalWhitePixels);

		float whatColor = totalRedPixels;
		int color = 1;
		//1-red; 2=brown; 3-yellow; 4-orange; 5-green; 6-white; 7- purple
		if (totalBrownPixels > whatColor)
		{
			whatColor = totalBrownPixels;
			color = 2;
			printf("Brown 11\n");
		}
		if (whatColor < totalGreenPixels)
		{
			whatColor = totalGreenPixels;
			color = 5;
		}
		if (whatColor < totalWhitePixels || totalWhitePixels > totalGreenPixels + 2000)
		{
			whatColor = totalWhitePixels;
			color = 6;
		}




		int dominantFoodColor = 0;


		//compare the total number of REDS VS BROWNS
		//then get the dominant color

		//RED: D-D-D-DOMINATING
		//if (totalRedPixels > totalBrownPixels)
		//	if (dominantFoodColor == 0)
		//	{
		//		printf("unclassified\n");
		//	}

		if (whatColor == totalRedPixels)
		{
			//GET ALL THE REDS CUZ YOU ARE SURE THE FOOOD IS SO RED
			//get half of food pixels
			float rednessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float rednessRange2 = foodAccumulate / 2;			//sample: 3000 range: 1500
			float rednessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total red pixels: %.0f\n", totalRedPixels);
			printf("redness range1: %.0f\n", rednessRange1);
			printf("redness range2: %.0f\n", rednessRange2);
			printf("redness range3: %.0f\n", rednessRange3);


			if (totalRedPixels > 900000 && totalRedPixels < 1200000)
			{
				printf("red 12 reached 1M red pixels, so offset value\n");
				//if total red pixels reaches 1Melyon 
				totalRedPixels -= 200000;								//deduct so it can fall to the last range :)
			}
			if (totalRedPixels > 1000 && totalRedPixels <= 300000)		//5000 because we dont want to catch the browns here too.
			{
				dominantFoodColor = 4;
				if (totalRedPixels > (rednessRange1 - 3500))
				{
					printf("red 13\n");
					dominantFoodColor = 1;
				}
			}

			if (totalRedPixels < rednessRange2 && totalRedPixels > rednessRange1)
			{
				printf("red 14\n");
				dominantFoodColor = 1;
			}
			//check if red pixels is half or more than half the total food pixels

			//check if red pixels is half or more than half the total food pixels
			if (totalRedPixels < rednessRange1 && (rednessRange3 - rednessRange2) <= 22000)   //orangyyyyy
			{
				printf("red 15\n");
				dominantFoodColor = 4;
				if (totalRedPixels >(rednessRange1 - 3500))
				{
					printf("red 16\n");
					// dominantFoodColor = 1;
				}
			}

			if (totalRedPixels < (rednessRange1 - 3500))
			{
				printf("red 17\n");
				dominantFoodColor = 1;
				//still orangy?

			}

			if (totalRedPixels < rednessRange3 && totalRedPixels > rednessRange2)
			{
				printf("red 18\n");
				dominantFoodColor = 1;
				//check for contours here that belong under the red category

			}
			else if (totalRedPixels > rednessRange3 && totalRedPixels < rednessRange3 + 100)
			{
				printf("red 19\n");
				dominantFoodColor = 1;
				//check for contours here that belong under the red category
			}

			else if (totalRedPixels > rednessRange3)
			{
				printf("red 20\n");
				dominantFoodColor = 1;
			}







			if (dominantFoodColor == 1)		//RED RED RED
			{
				if (totalRedPixels > rednessRange3 + 50)
				{
					printf("red 21\n");
					dominantFoodColor = 7;	//purple					
				}

				else
				{


					printf("red 22\n");
					printf("soooooo Red\n");
					char str3[] = "fruit";

					//assuming that anything red is an apple for now...



					char calorieCount[50];
					float oneExchange = 86.0;			//these are for apple only
					float carbs = 10.0;
					float protein = 0.0;
					float fat = 0.0;
					float appleCalories = 40.0;
					float totalGramsDivOneExchange = 0;


					totalGramsDivOneExchange = totalGrams / oneExchange;
					carbs = carbs * totalGramsDivOneExchange;
					protein *= totalGramsDivOneExchange;
					appleCalories = appleCalories * totalGramsDivOneExchange;


					cv::Point textOrgV11(bounding_rect.x, bounding_rect.y);
					putText(matSrc, "RED FOOD", textOrgV11, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					cv::Point textOrgV1(bounding_rect.x, bounding_rect.y + 30);
					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					putText(matSrc, "Apple grams:" + strTotalGrams + "grams", textOrgV1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();
					cv::Point textOrgV2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "Apple carbohydrates: " + strCarbs + "calories", textOrgV2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strP;
					strP << protein;
					std::string strProtein = strP.str();
					cv::Point textOrgV3(bounding_rect.x, bounding_rect.y + 90);
					putText(matSrc, "Apple  Protein: " + strProtein + "calories", textOrgV3, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strF;
					strF << fat;
					std::string strFat = strF.str();
					cv::Point textOrgV4(bounding_rect.x, bounding_rect.y + 120);
					putText(matSrc, "Apple Fat: " + strFat + "calories", textOrgV4, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strCV;
					strCV << appleCalories;
					std::string strCaloriesVegie = strCV.str();
					cv::Point textOrgV5(bounding_rect.x, bounding_rect.y + 150);
					putText(matSrc, "Apple Total Calories: " + strCaloriesVegie + "calories", textOrgV5, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					printf("it can be one of the following: cherry\nbeef tenderloin\nbeefshank\nbreastmeat pitso\ncarabeefshank\nleanpork\napple");

				}

			}



			//-----------------
			if (dominantFoodColor == 4)
			{

				printf("it is a carrot or anything orange\n");


				//insert contour algorithm here in order to know what kind of orange food this is 

				char calorieCount[50];
				float oneExchange = 25.0;	//if one exchange is 25 or less, group A, else it's group B.
				float carbs = 0.0;
				float protein = 0.0;
				float fat = 0.0;
				float vegieCalories = 0.0;
				float totalGramsDivOneExchange = 0;

				if (totalGrams <= oneExchange)	//computation for group A
				{
					carbs *= 0.0;
					protein *= 0.0;
					fat *= 0.0;
				}
				if (totalGrams > oneExchange)	//computation for group B
				{
					carbs = 3.0;
					protein = 1.0;
					vegieCalories = 16.0;
					oneExchange = 40.0;
					totalGramsDivOneExchange = totalGrams / oneExchange;
					carbs = carbs * totalGramsDivOneExchange;
					protein *= totalGramsDivOneExchange;
					vegieCalories = vegieCalories * totalGramsDivOneExchange;
				}

				cv::Point textOrgV11(bounding_rect.x, bounding_rect.y);
				putText(matSrc, "ORANGE FOOD", textOrgV11, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				cv::Point textOrgV1(bounding_rect.x, bounding_rect.y + 30);
				std::ostringstream strT;
				strT << totalGrams;
				std::string strTotalGrams = strT.str();
				putText(matSrc, "Ototal grams:" + strTotalGrams + "grams", textOrgV1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				std::ostringstream strC;
				strC << carbs;
				std::string strCarbs = strC.str();
				cv::Point textOrgV2(bounding_rect.x, bounding_rect.y + 60);
				putText(matSrc, "Ocarbohydrates: " + strCarbs + "calories", textOrgV2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strP;
				strP << protein;
				std::string strProtein = strP.str();
				cv::Point textOrgV3(bounding_rect.x, bounding_rect.y + 90);
				putText(matSrc, "OProtein: " + strProtein + "calories", textOrgV3, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strF;
				strF << fat;
				std::string strFat = strF.str();
				cv::Point textOrgV4(bounding_rect.x, bounding_rect.y + 120);
				putText(matSrc, "OFat: " + strFat + "calories", textOrgV4, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strCV;
				strCV << vegieCalories;
				std::string strCaloriesVegie = strCV.str();
				cv::Point textOrgV5(bounding_rect.x, bounding_rect.y + 150);
				putText(matSrc, "OTotal Calories: " + strCaloriesVegie + "calories", textOrgV5, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				//-------------------------
			}



			if (dominantFoodColor == 7)
			{
				printf("soooooo purple\n");
				char str3[] = "fruit";


				char calorieCount[50];
				float oneExchange = 140.0;
				float carbs = 10.0;
				float kcal = 40.0;					//1 exchange = 10 grams carbs = 40kcal
				//let's say totalgrams = 112;		//generated by the system
				if (totalGrams > oneExchange)
				{
					totalGrams /= oneExchange;
					carbs = carbs * totalGrams;
					kcal = carbs * totalGrams;
				}
				//				sprintf_s(calorieCount, "%s has %.0f grams.\n%.0f grams of Carbs = %.0f kcal\n", str3, totalGrams, carbs, kcal);


				cv::Point textOrg(bounding_rect.x, bounding_rect.y);
				putText(matSrc, str3, textOrg, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strT;
				strT << totalGrams;
				std::string strTotalGrams = strT.str();
				cv::Point textOrg1(bounding_rect.x, bounding_rect.y + 30);
				putText(matSrc, "total grams:" + strTotalGrams + "grams", textOrg1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strC;
				strC << carbs;
				std::string strCarbs = strC.str();

				cv::Point textOrg2(bounding_rect.x, bounding_rect.y + 60);
				putText(matSrc, "carbohydrates: " + strCarbs + "calories", textOrg2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				printf("can be grapes or anything purple\n");

			}






			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat redHistoimage_resize;
			cv::resize(redHistoImage, redHistoimage_resize, cv::Size2i(redHistoImage.cols / 2, redHistoImage.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("redHhistoImageResizeW");
			cv::imshow("redhistoImageResizeW", redHistoimage_resize);

			cv::imwrite("redhistogram.png", redHistoimage_resize);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("red Histogram");
			cv::imshow("red Histogram", h.getHistogramImage(redHistoimage_resize));
			//****** END OF DRAW ZE coin HISTOGRAM 




			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat red_thresholded;
			cv::threshold(redHistoimage_resize, red_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("red binaryImage");
			cv::imshow("red binaryImage", red_thresholded);

			//end of getting the RED histogram
			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

			//get histogram and total number of pixels of black and white of cropped food
			//it is considered red if total pixels of "red"()




			//resize and display
			cv::Mat lower_red_hue_range_reduced, upper_red_hue_range_reduced, red_hue_image_reduced;
			cv::resize(lower_red_hue_range, lower_red_hue_range_reduced, cv::Size2i(lower_red_hue_range.cols / 2, lower_red_hue_range.rows / 2));
			imshow("lower_red_hue_range_reduced", lower_red_hue_range_reduced);

			cv::resize(upper_red_hue_range, upper_red_hue_range_reduced, cv::Size2i(upper_red_hue_range.cols / 2, upper_red_hue_range.rows / 2));
			imshow("upper_red_hue_range_reduced", upper_red_hue_range_reduced);


			cv::resize(red_hue_image, red_hue_image_reduced, cv::Size2i(red_hue_image.cols / 2, red_hue_image.rows / 2));
			imshow("red_hue_image_reduced", red_hue_image_reduced);


			//END OF GETTING ALL THE REDS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------

		}


		//compare the total number of REDS VS BROWNS vs GREENS
		//then get the dominant color

		//BROWN: D-D-D-DOMINATING
		//else if (totalBrownPixels > totalRedPixels)
		else if (whatColor == totalBrownPixels || color == 2)
		{

			//GET ALL THE REDS CUZ YOU ARE SURE THE FOOOD IS SO RED
			//get half of food pixels
			float brownnessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float brownnessRange2 = (foodAccumulate / 2) + 1000;			//sample: 3000 range: 1500
			float brownnessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total brown pixels: %.0f\n", totalBrownPixels);
			printf("brownness range1: %.0f\n", brownnessRange1);
			printf("brownness range2: %.0f\n", brownnessRange2);
			printf("brownness range3: %.0f\n", brownnessRange3);



			float oneThirdThebrownness = foodAccumulate / 1.5;		//it can be yellow			
			float halfTheBrownness = foodAccumulate / 2.0;



			if (totalRedPixels > 1000 && totalRedPixels <= 200000 && !(totalBrownPixels > brownnessRange2 && totalBrownPixels < brownnessRange3))		//5000 because we dont want to catch the browns here too.
			{
				dominantFoodColor = 4;		//4 is for orange :)	
				printf("Brown orange\n");
			}

			else if (totalBrownPixels >= brownnessRange2 && totalBrownPixels <= 300000)	//because 300k is yellow already
			{
				dominantFoodColor = 3;
				printf("Brown yellow\n");
			}

			else if (totalBrownPixels < brownnessRange2 && totalBrownPixels > brownnessRange1)	//singkamas brown
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 12\n");
			}
			else if (totalBrownPixels > brownnessRange2 && totalBrownPixels <= brownnessRange3)
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 14\3n");
			}

			else if (totalBrownPixels <= halfTheBrownness)
			{
				dominantFoodColor = 2;
				color = 2;
				printf("Brown 15\n");

				if ((brownnessRange3 - brownnessRange2) > 20000 && (brownnessRange3 - brownnessRange2) < 30000)		//this will be considered yellow-ish already
				{
					dominantFoodColor = 3;
					color = 3;
					printf("Brown yelloooow\n");
				}

			}

			else if (dominantFoodColor == 4)
			{
				char str3[] = "brownish Orange";
				putText(matSrc, str3, center, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);
				printf("this can be orange\ncarrots\ntomato\nchico\nor anything brownish orange\n");
			}



			if (color == 2 || dominantFoodColor == 2)
			{


				char str3[] = "Bread?";


				int halfOfBrownRange2And3 = brownnessRange2 + ((brownnessRange3 - brownnessRange2) * 0.5);
				//get the half of range 2 and range 3

				if (totalBrownPixels < brownnessRange2 && totalBrownPixels > brownnessRange1)
				{
					printf("Brown 16\n"); // singkamas brown
				}

				else if (totalBrownPixels < halfOfBrownRange2And3)
				{
					printf("THIS IS BROWN\n");
					printf("so far, camote, chico kind of brown");



					//use the contour code here to recognize what brown food it is. and use the calorie formula for that food 




					//chico formula, use this!					
					char calorieCount[50];
					float oneExchange = 54.0;
					float carbs = 0.0;
					float protein = 0.0;
					float fat = 0.0;
					float brownCalories = 40.0;
					float totalGramsDivOneExchange = 0;


					totalGramsDivOneExchange = totalGrams / oneExchange;
					carbs = carbs * totalGramsDivOneExchange;
					protein *= totalGramsDivOneExchange;
					brownCalories = brownCalories * totalGramsDivOneExchange;


					cv::Point textOrgV11(bounding_rect.x, bounding_rect.y);
					putText(matSrc, "BROWN ______", textOrgV11, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					cv::Point textOrgV1(bounding_rect.x, bounding_rect.y + 30);
					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					putText(matSrc, "Btotal grams:" + strTotalGrams + "grams", textOrgV1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();
					cv::Point textOrgV2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "Bcarbohydrates: " + strCarbs + "calories", textOrgV2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strP;
					strP << protein;
					std::string strProtein = strP.str();
					cv::Point textOrgV3(bounding_rect.x, bounding_rect.y + 90);
					putText(matSrc, "BProtein: " + strProtein + "calories", textOrgV3, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strF;
					strF << fat;
					std::string strFat = strF.str();
					cv::Point textOrgV4(bounding_rect.x, bounding_rect.y + 120);
					putText(matSrc, "BFat: " + strFat + "calories", textOrgV4, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strCV;
					strCV << brownCalories;
					std::string strCaloriesBrown = strCV.str();
					cv::Point textOrgV5(bounding_rect.x, bounding_rect.y + 150);
					putText(matSrc, "BTotal Calories: " + strCaloriesBrown + "calories", textOrgV5, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					printf("this food is so camote brown\n");



					//but follow vegetable formula and display if it is a vegetable.

				}



				if (totalBrownPixels > halfOfBrownRange2And3)
				{
					printf("Brown 17\n");
				}


			}
			else if (dominantFoodColor == 3)
			{
				char str3[] = "Mango";

				//char str3[] = "fruit";

				int halfOfBrownRange2And3 = brownnessRange2 + ((brownnessRange3 - brownnessRange2) * 0.5);
				//get the half of range 2 and range 3

				if (totalBrownPixels > halfOfBrownRange2And3)
				{
					dominantFoodColor = 2;
					color = 2;
					printf("Brown 18\n");
				}

				else if (totalBrownPixels < halfOfBrownRange2And3)
				{

					//THIS IS SO YELLOW



					//code for knowing what yellow food it is here.




					//assuming that everything yellow here is a banana (lakatan - that long yellow banana )
					printf("This food is so yellow\n");

					char calorieCount[50];
					float oneExchange = 51.0;	//if one exchange is 25 or less, group A, else it's group B.
					float carbs = 10.0;
					float protein = 0.0;
					float fat = 0.0;
					float yellowCalories = 40.0;
					float totalGramsDivOneExchange = 0;



					totalGramsDivOneExchange = totalGrams / oneExchange;
					carbs = carbs * totalGramsDivOneExchange;
					protein *= totalGramsDivOneExchange;
					yellowCalories = yellowCalories * totalGramsDivOneExchange;



					cv::Point textOrgV11(bounding_rect.x, bounding_rect.y);
					putText(matSrc, "YELLOW FOOD", textOrgV11, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					cv::Point textOrgV1(bounding_rect.x, bounding_rect.y + 30);
					std::ostringstream strT;
					strT << totalGrams;
					std::string strTotalGrams = strT.str();
					putText(matSrc, "Yellow total grams:" + strTotalGrams + "grams", textOrgV1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


					std::ostringstream strC;
					strC << carbs;
					std::string strCarbs = strC.str();
					cv::Point textOrgV2(bounding_rect.x, bounding_rect.y + 60);
					putText(matSrc, "Yellow carbohydrates: " + strCarbs + "calories", textOrgV2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strP;
					strP << protein;
					std::string strProtein = strP.str();
					cv::Point textOrgV3(bounding_rect.x, bounding_rect.y + 90);
					putText(matSrc, "Yellow Protein: " + strProtein + "calories", textOrgV3, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strF;
					strF << fat;
					std::string strFat = strF.str();
					cv::Point textOrgV4(bounding_rect.x, bounding_rect.y + 120);
					putText(matSrc, "Yellow Fat: " + strFat + "calories", textOrgV4, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

					std::ostringstream strCV;
					strCV << yellowCalories;
					std::string strCaloriesVegie = strCV.str();
					cv::Point textOrgV5(bounding_rect.x, bounding_rect.y + 150);
					putText(matSrc, "VTotal Calories: " + strCaloriesVegie + "calories", textOrgV5, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);







					printf("this food is so yellow\n");
					printf("it can be one of the following: banana and everything yellow");
					//check for contours here that belong under the red category
				}

			}


			//--


			//std::stringstream ss;
			//ss << redAccumulate;
			//std::string s = ss.str();

			//then you can pass s to PutText


			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat brownHistoimage_resize;
			cv::resize(brownHistoImage, brownHistoimage_resize, cv::Size2i(brownHistoImage.cols / 2, brownHistoImage.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("brownHhistoImageResizeW");
			cv::imshow("brownHhistoImageResizeW", brownHistoimage_resize);

			cv::imwrite("brownhistogram.png", brownHistoimage_resize);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("brown Histogram");
			cv::imshow("brown Histogram", h.getHistogramImage(brownHistoimage_resize));
			//****** END OF DRAW ZE coin HISTOGRAM 



			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat brown_thresholded;
			cv::threshold(brownHistoimage_resize, brown_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("brown binaryImage");
			cv::imshow("brown binaryImage", brown_thresholded);

			//end of getting the RED histogram
			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(brown_hue_image, brown_hue_image, cv::Size(9, 9), 2, 2);


			//get histogram and total number of pixels of black and white of cropped food	
			//resize and display
			cv::Mat lower_brown_hue_range_reduced, upper_brown_hue_range_reduced, brown_hue_image_reduced;
			cv::resize(lower_brown_hue_range, lower_brown_hue_range_reduced, cv::Size2i(lower_brown_hue_range.cols / 2, lower_brown_hue_range.rows / 2));
			imshow("lower_brown_hue_range_reduced", lower_brown_hue_range_reduced);

			cv::resize(upper_brown_hue_range, upper_brown_hue_range_reduced, cv::Size2i(upper_brown_hue_range.cols / 2, upper_brown_hue_range.rows / 2));
			imshow("upper_brown_hue_range_reduced", upper_brown_hue_range_reduced);


			cv::resize(brown_hue_image, brown_hue_image_reduced, cv::Size2i(brown_hue_image.cols / 2, brown_hue_image.rows / 2));
			imshow("brown_hue_image_reduced", brown_hue_image_reduced);

			//END OF GETTING ALL THE BROWNS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------

		}






		//GREEN: D-D-D-DOMINATING
		if (whatColor == totalGreenPixels)
		{
			dominantFoodColor = 5;	//1 is for red
			printf("It's a vegetable C\n");
			//GET ALL THE GREENS CUZ YOU ARE SURE THE FOOOD IS SO RED
			//get half of food pixels
			float greennessRange1 = foodAccumulate / 2.8;			//sample: 3000 range: 1000
			float greennessRange2 = foodAccumulate / 2;			//sample: 3000 range: 1500
			float greennessRange3 = foodAccumulate / 1.5;			//sample: 3000 range: 2000


			printf("total green pixels: %.0f\n", totalGreenPixels);
			printf("Greenness range1: %.0f\n", greennessRange1);
			printf("Greenness range2: %.0f\n", greennessRange2);
			printf("Greenness range3: %.0f\n", greennessRange3);

			if (totalGreenPixels > 900000 && totalGreenPixels < 1200000)
			{

				//if total green pixels reaches 1Melyon 
				totalGreenPixels -= 200000;								//deduct so it can fall to the last range :)
			}
			if (totalGreenPixels > 1000 && totalGreenPixels <= 300000)		//5000 because we dont want to catch the browns here too.
			{
				printf("it is a vegetable D\n");
				dominantFoodColor = 5;
			}

			//check if red pixels is half or more than half the total food pixels
			else if (totalGreenPixels > 300000 && totalGreenPixels <= greennessRange3)
			{
				printf("it is a vegetable E\n");
				dominantFoodColor = 5;
				//check for contours here that belong under the red category
			}
			//check if red pixels is half or more than half the total food pixels
			else if (totalGreenPixels <= greennessRange3 || totalGreenPixels < greennessRange3)
			{
				printf("it is a vegetable F\n");
				dominantFoodColor = 5;
				//check for contours here that belong under the green category
			}



			//std::stringstream ss;
			//ss << redAccumulate;
			//std::string s = ss.str();

			//then you can pass s to PutText




			//resize the original img, as half size of original cols and rows
			//cv::Size2i(img.cols/2, img.rows/2) is where you specify the desired dimension for resized image
			cv::Mat greenHistoimage_resize;
			cv::resize(greenHistoImage, greenHistoimage_resize, cv::Size2i(greenHistoImage.cols / 2, greenHistoImage.rows / 2));

			//visualize the original and resized image in window: lena and lena_size respectively.	
			cv::namedWindow("greenHhistoImageResizeW");
			cv::imshow("greenhistoImageResizeW", greenHistoimage_resize);

			cv::imwrite("greenhistogram.png", greenHistoimage_resize);


			//****** DRAW ZE HISTOGRAM 
			cv::namedWindow("green Histogram");
			cv::imshow("green Histogram", h.getHistogramImage(greenHistoimage_resize));
			//****** END OF DRAW ZE coin HISTOGRAM 




			//thresholding to create a binary image
			//here we threshold the image at the minimum value just before the
			//increase toward the high peak of the histogram (gray value 102)
			cv::Mat green_thresholded;
			cv::threshold(greenHistoimage_resize, green_thresholded, 84, 130, cv::THRESH_BINARY);
			cv::namedWindow("green binaryImage");
			//cv::imshow("green binaryImage", green_thresholded);

			//end of getting the RED histogram
			//------------------------------------------------------------------------
			//Then blur it :)
			cv::GaussianBlur(green_hue_image, green_hue_image, cv::Size(9, 9), 2, 2);

			//get histogram and total number of pixels of black and white of cropped food
			//it is considered red if total pixels of "red"()




			//resize and display
			cv::Mat lower_green_hue_range_reduced, upper_green_hue_range_reduced, green_hue_image_reduced;
			cv::resize(lower_green_hue_range, lower_green_hue_range_reduced, cv::Size2i(lower_green_hue_range.cols / 2, lower_green_hue_range.rows / 2));
			imshow("lower_green_hue_range_reduced", lower_green_hue_range_reduced);

			cv::resize(upper_green_hue_range, upper_green_hue_range_reduced, cv::Size2i(upper_green_hue_range.cols / 2, upper_green_hue_range.rows / 2));
			imshow("upper_green_hue_range_reduced", upper_green_hue_range_reduced);


			cv::resize(green_hue_image, green_hue_image_reduced, cv::Size2i(green_hue_image.cols / 2, green_hue_image.rows / 2));
			imshow("green_hue_image_reduced", green_hue_image_reduced);


			//END OF GETTING ALL THE GREENS
			//----------------------------------------------------------------------------------------------------------------------------------------------------------------








			if (dominantFoodColor == 5)
			{

				//place here the code that will compare contours and determine what green food it is..







				printf("it is a vegetable G\n");

				//Group A vegetables contain negligible carbohydrates, protein and energy if 1 exchange or less is used. 
				//When 2 exchanges are used, compute as one Group B Vegetables. 
				//The portion size for one exchange is
				//Vegetable A: 1 exchange = 1cup raw (25g) or 1/2 cup cooked (45g); 
				//Vegetable B: 1 exchange = 1/2 cup raw (40g) or 1/2 cup cooked (45g)

				char calorieCount[50];
				float oneExchange = 25.0;	//if one exchange is 25 or less, group A, else it's group B.
				float carbs = 0.0;
				float protein = 0.0;
				float fat = 0.0;
				float vegieCalories = 0.0;
				float totalGramsDivOneExchange = 0;

				if (totalGrams <= oneExchange)	//computation for group A
				{
					carbs *= 0.0;
					protein *= 0.0;
					fat *= 0.0;
				}
				if (totalGrams > oneExchange)	//computation for group B
				{
					carbs = 3.0;
					protein = 1.0;
					vegieCalories = 16.0;
					oneExchange = 40.0;
					totalGramsDivOneExchange = totalGrams / oneExchange;
					carbs = carbs * totalGramsDivOneExchange;
					protein *= totalGramsDivOneExchange;
					vegieCalories = vegieCalories * totalGramsDivOneExchange;
				}

				cv::Point textOrgV11(bounding_rect.x, bounding_rect.y);
				putText(matSrc, "VEGETABLES", textOrgV11, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				cv::Point textOrgV1(bounding_rect.x, bounding_rect.y + 30);
				std::ostringstream strT;
				strT << totalGrams;
				std::string strTotalGrams = strT.str();
				putText(matSrc, "Vtotal grams:" + strTotalGrams + "grams", textOrgV1, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);


				std::ostringstream strC;
				strC << carbs;
				std::string strCarbs = strC.str();
				cv::Point textOrgV2(bounding_rect.x, bounding_rect.y + 60);
				putText(matSrc, "Vcarbohydrates: " + strCarbs + "calories", textOrgV2, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strP;
				strP << protein;
				std::string strProtein = strP.str();
				cv::Point textOrgV3(bounding_rect.x, bounding_rect.y + 90);
				putText(matSrc, "VProtein: " + strProtein + "calories", textOrgV3, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strF;
				strF << fat;
				std::string strFat = strF.str();
				cv::Point textOrgV4(bounding_rect.x, bounding_rect.y + 120);
				putText(matSrc, "VFat: " + strFat + "calories", textOrgV4, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

				std::ostringstream strCV;
				strCV << vegieCalories;
				std::string strCaloriesVegie = strCV.str();
				cv::Point textOrgV5(bounding_rect.x, bounding_rect.y + 150);
				putText(matSrc, "VTotal Calories: " + strCaloriesVegie + "calories", textOrgV5, FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 0, 0), 1, CV_AA);

			}










		}

	} //endof for (size_t i = 0; i < circles.size(); i++)


	//printf("coin size: %d\n", copyOfCoin.size());								//CALCULATE GRAMS OF FOOD HERE
	//printf("matSrc size: %d\n", matSrc.size());														//TOTAL NUMBER OF PIXELS IN CHOSEN FOOD






	//another batch of image reduction
	cv::Mat matSrc_reduced;
	cv::resize(matSrc, matSrc_reduced, cv::Size2i(matSrc.cols / 2, matSrc.rows / 2));


























	//-------------------------------------------------------------
	//
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[6] = { -1.0, 1.0, -1.0, 1.0, 1.0, 1.0 };
	Mat labelsMat(5, 1, CV_32FC1, labels);

	float trainingData[6][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 }, { 50, 10 }, { 5, 10 } };
	Mat trainingDataMat(5, 2, CV_32FC1, trainingData);




	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(5, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(5, 10), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}



	//	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	imshow("Training Data Mat", trainingDataMat);
	imshow("Labels", labelsMat);

	// Data for visual representation ends here
	//
	//-------------------------------------------------------------




	//MACHINE LEARNING ENDS RIGHT HERE
	//----------------------------------------------------------------------------------------------------------------------------------------------------------------





	//show results
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	imshow("Original Image", reduced_src);

	namedWindow("Final Output", CV_WINDOW_AUTOSIZE);
	imshow("Final Output", matSrc_reduced);

	namedWindow("canny", CV_WINDOW_AUTOSIZE);
	imshow("canny", matCanny_reduced);

	namedWindow("mat rgb canny", CV_WINDOW_AUTOSIZE);
	imshow("mat rgb canny", matRgbCanny_reduced);

	namedWindow("matBinary2", CV_WINDOW_AUTOSIZE);
	imshow("matBinary2", matBinary_reduced);


	namedWindow("contours", CV_WINDOW_AUTOSIZE);			//may be omitted
	imshow("contours", matDrawing_reduced);

	namedWindow("no background", CV_WINDOW_AUTOSIZE);
	imshow("no background", nobackground_reduced);

	namedWindow("in HSV version", CV_WINDOW_AUTOSIZE);
	imshow("in HSV version", matImHSV_reduced);


	cvNamedWindow("HSV window", 1);
	imshow("HSV window", matImgHSV_reduced);

	cvNamedWindow("Food Contour", 1);
	imshow("Food Contour", binaryHSV_reduced);
	cvNamedWindow("Final", 1);
	imshow("Final", matImgHSV_reduced);











	waitKey(-1);

	destroyAllWindows();
	cvDestroyWindow("in HSV version");
	cvDestroyWindow("Original Image");
	cvDestroyWindow("Circle Detection1");
	cvDestroyWindow("canny");
	cvDestroyWindow("mat rgb canny");
	cvDestroyWindow("matGrrr");
	cvDestroyWindow("binary");
	cvDestroyWindow("matBinary2");
	cvDestroyWindow("contours");
	cvDestroyWindow("foreground");
	cvDestroyWindow("Training Data");
	cvDestroyWindow("Training Data Histogram");
	cvDestroyWindow("Food Isolated");
	//do stuff here
	cvReleaseImage(&imHSV);			//frees the new HSV image
	cvReleaseImage(&imRGB);			//frees the original RGB image

	//	KNearest knn(trainingVectors, trainingLabels);


	fs.release();
	fclose(fp);
	return 0;
}