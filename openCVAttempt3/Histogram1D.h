#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2\imgproc\imgproc.hpp>	//required for any image related functions


using namespace cv;

class Histogram1D
{
private:
	int histSize[1]; //number of bins
	float hranges[2]; //min and max pixel value
	const float* ranges[1];
	int channels[1]; //only 1 channel used here
public:
	Histogram1D();

	cv::MatND getHistogram(const cv::Mat &image){
		cv::MatND hist;

		//compute histogram
		cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
		//1 - histogram form 1 image only
		//channels - the channel used
		//cv::Mat() - no mask is used
		//hist - the resulting histogram
		//1 - it is a 1D histogram
		//histSize - number of bins
		//ranges - pixel value range

		return hist;
	}

	cv::Mat getHistogramImage(const cv::Mat &image){
		//compute histogram first
		cv::MatND hist = getHistogram(image);

		//get min and max bin values
		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		//image on which to display histogram
		cv::Mat histImg(histSize[0], histSize[0], CV_8U, cv::Scalar(255));

		//set highest point at 90% of nbins
		int hpt = static_cast<int>(0.9*histSize[0]);

		//draw a vertical line for each bin
		for (int h = 0; h < histSize[0]; h++){
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt / maxVal);

			//this function draws a line between two points
			cv::line(histImg, cv::Point(h, histSize[0]), cv::Point(h, histSize[0] - intensity), cv::Scalar::all(0));
		}
		return histImg;
	}

	//image - input image, lookup - 1x256 matrix
	cv::Mat applyLookUp(const cv::Mat& image, const cv::Mat& lookup){
		//the output image						
		cv::Mat result;							//how to create YOUR OWN inversion table page 108 
		//apply lookup table
		cv::LUT(image, lookup, result);
		return result;
	}

	cv::Mat stretch(const cv::Mat &image, int minValue = 0){
		//compute histogram first
		cv::MatND hist = getHistogram(image);
		//find left extremity of the histogram
		int imin = 0;
		for (; imin < histSize[0]; imin++){
			std::cout << hist.at<float>(imin) << std::endl;
			if (hist.at<float>(imin) > minValue)
				break;
		}
		//find right extremity of the histogram
		int imax = histSize[0] - 1;
		for (; imax >= 0; imax--){
			if (hist.at<float>(imax) > minValue)
				break;
		}

		//create lookup table
		int dim(256);
		//1 Dimension, 256 entries, uchar
		cv::Mat lookup(1, &dim, CV_8U);
		//build lookup table
		for (int i = 0; i < 256; i++){
			//stretch between imin and imax
			if (i < imin) lookup.at<uchar>(i) = 0;
			else if (i > imax) lookup.at<uchar>(i) = 255;
			//linear mapping
			else lookup.at<uchar>(i) = static_cast<uchar>(255.0*(i - imin) / (imax - imin) + 0.5);
		}
		//apply lookup table
		cv::Mat result;
		result = applyLookUp(image, lookup);
		return result;
	}


};

