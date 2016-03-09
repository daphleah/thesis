#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2\imgproc\imgproc.hpp>	//required for any image related functions

using namespace cv;


class ColorHistogram
{
private:
	int histSize[3];
	float hranges[2];
	const float*ranges[3];
	int channels[3];
public:
	ColorHistogram();

	

	cv::MatND getHistogram(const cv::Mat &image){
		cv::MatND hist;
		//compute histogram
		cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
		return hist;
	}

	cv::SparseMat getSparseHistogram(const cv::Mat &image){
		cv::SparseMat hist(3, histSize, CV_32F);
		//compute histogram
		cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
		return hist;
	}

	void colorReduce(cv::Mat&image, int div = 64){
		int nl = image.rows;	//number of lines
		int nc = image.cols;	//number of columns

		//is it a continuous image?
		if (image.isContinuous()){
			//tehn no padded pixels
			nc = nc*nl;
			nl = 1;	//it is now a 1D array
		}
		int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
		//mask used to round the pixel value
		uchar mask = 0xFF << n;	//e.g.for div=16 mask=0xF0
		//for all pixels
		for (int j = 0; j < nl; j++){
			//pointer to first column of line j
			uchar* data = image.ptr<uchar>(j);
			for (int i = 0; i < nc; i++){
				//process each pixel
				*data++ = *data&mask + div / 2;
				*data++ = *data&mask + div / 2;
				*data++ = *data&mask + div / 2;
				//end of pixel processing
			}//end of line
		}
	}
};

