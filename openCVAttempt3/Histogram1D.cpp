#include "Histogram1D.h"


Histogram1D::Histogram1D()
{
	//prepare arguments for 1D histogram
	histSize[0] = 256;
	hranges[0] = 0.0;
	hranges[1] = 255.0;
	ranges[0] = hranges;
	channels[0] = 0; //by default, we look at channel 0
}
