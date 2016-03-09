#include "ColorHistogram.h"


ColorHistogram::ColorHistogram(){
	//prepare arguments for a color histogram
	histSize[0] = histSize[1] = histSize[2] = 256;
	hranges[0] = 0.0;	//BRG image
	hranges[1] = 255.0;
	ranges[0] = hranges;	//all channels have the same range
	ranges[1] = hranges;
	ranges[2] = hranges;
	channels[0] = 0;	//the three channels
	channels[1] = 1;
	channels[2] = 2;
}