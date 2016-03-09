#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2\imgproc\imgproc.hpp>	//required for any image related functions

using namespace cv;

class ContentFinder
{
private:
	float hranges[2];
	const float* ranges[3];
	int channels[3];
	float threshold;
	cv::MatND histogram;

public:
	ContentFinder();

	void setThreshold(float t){
		threshold = t;
	}
	//gets the threshold()
	float getThreshold(){
		return threshold;
	}
	//Sets the reference histogram
	void setHistogram(const cv::MatND& h){
		histogram = h;
		cv::normalize(histogram, histogram, 1.0);
	}
	
	cv::Mat find(const cv::Mat& image, float minValue, float maxValue, int *channels, int dim){
		cv::Mat result;
		hranges[0] = minValue;
		hranges[1] = maxValue;
		for (int i = 0; i < dim; i++)
			this->channels[i] = channels[i];
		cv::calcBackProject(&image, 1, channels, histogram, result, ranges, 255.0);

		if (threshold > 0.0)
			cv::threshold(result, result, 255 * threshold, 255, cv::THRESH_BINARY);
		return result;
	}
	
};

