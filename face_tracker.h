#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "face.h"


// Face tracker will keep track of the faces that have been seen and attempt to
// track them from frame to frame. There is a high false detection rate by the 
// face classifier at times, so we need to track stuff for a while before saying it's
// actually a real face.
//
// Faces are first in the "potentialFaces" list, in which they are not masked, but
// become masked as soon as they are promoted to "definiteFaces"
//
// This is similiar to how radar/video trackers decide if an object is real and to 
// start tracking it.
class FaceTracker {

    // Remove a face if it hasn't been seen for 15 frames
    static constexpr int EXPIRE_FACE_FRAMES = 15;

    // Require being seen 10 times before considered real
    static constexpr int DETECTIONS_REQUIRED_TO_BE_REAL = 8;

    // Keep track of potential and real faces
    std::vector<Face> definiteFaces;
    std::vector<Face> potentialFaces;
    
    // Initialize possible mask set for each face
    MaskVector masks;
    
    public:

        FaceTracker();

        // Take face detections from a frame and insert to the face tracker
        void processNewDetections(const std::vector<cv::Rect> detections, const int& nFrame);

        // To draw what we have
        const std::vector<Face>& getFaces() const;
    
    private:

        // Need to run stats on size and position similarity to check for a match
        bool matchesFace(const cv::Rect & potential, Face & face, const int& nFrame);

        // Check detections against a std::vector of faces for matches, and return
        // true if a match is found. 
        //
        // FIXME - future improvement to do a nearest neighbor check than return
        // first match
        bool matchesAnyFace(const cv::Rect& detection, std::vector<Face>& faceVector, const int& nFrame, bool real = false);

        // Purge things that have not been seen in a while
        void purgeOldFaces(const int& nFrame);

        // Move faces from potential -> definite
        void upgradePotentialFaces();
        
};
