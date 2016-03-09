#include "ContentFinder.h"


ContentFinder::ContentFinder() : threshold(-1.0f){
	ranges[0] = hranges;	//all channels have same range
	ranges[1] = hranges;
	ranges[2] = hranges;
}
