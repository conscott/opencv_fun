#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "face.h"

// Start ID at 0
int Face::numFaces = 0;

Face::Face(const cv::Rect& detection, Mask& _mask, const int& nFrame) : mask(_mask), lastPosition(detection), lastFrameSeen(nFrame)  {
    lastTimeSeen = (double)cv::getTickCount();
    numDetections = 1;
    numFaces += 1;
    id = numFaces;
}

// Time pasted since last detection in milliseconds
double Face::timeUndetectedMs() const {
    return ((double)cv::getTickCount() - this->lastTimeSeen)*1000 / cv::getTickFrequency();
}

// Number of frames since last detection
int Face::undetectedFrames(const int& currentFrame) const {
    return currentFrame - lastFrameSeen;
}

// Update members with latest information
void Face::updateSeen(const cv::Rect & detection, const int& nFrame) {
    lastPosition = detection;
    lastTimeSeen = (double)cv::getTickCount();
    numDetections += 1;
    lastFrameSeen = nFrame;
    std::cout << "Updating face " << id << ": " << numDetections << " detections" << std::endl;
}

const Mask& Face::getMask() const {
    return mask;
}
