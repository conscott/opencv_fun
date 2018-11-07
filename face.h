#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "mask.h"


// A face we are tracking in the image, along with some stastics about
// its previous position, size, and detection stats
class Face 
{

    private:
        // Will be used to give new faces id's
        static int numFaces;

    public: 
        // To overlay on face
        Mask mask;
        // Last position seen
        cv::Rect lastPosition;
        // Time last seen
        double lastTimeSeen;
        // Last frame number seen
        int lastFrameSeen;
        // Total number of detections
        int numDetections;
        // Unique id for object
        int id;


        Face(const cv::Rect& detection, Mask& _mask, const int& nFrame);

        // Time pasted since last detection in milliseconds
        double timeUndetectedMs() const;
        
        // Number of frames since last detection
        int undetectedFrames(const int& currentFrame) const;
        
        // Update members with latest information
        void updateSeen(const cv::Rect & detection, const int& nFrame);

        const Mask& getMask() const;

};
