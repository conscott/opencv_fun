#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "face_tracker.h"

// Just needs to seed the default mask list
FaceTracker::FaceTracker() {
    masks = MaskVector();
    masks.emplace_back("imgs/hair2.png", 1.5);
    masks.emplace_back("imgs/guy3.png", 1.3);
    masks.emplace_back("imgs/glasses.png", 1.3);
    masks.emplace_back("imgs/ball.png", 1.5);
}


// Take face detections from a frame and insert to the face tracker
void FaceTracker::processNewDetections(const std::vector<cv::Rect> detections, const int& nFrame) {

    std::cout << "Processing " << detections.size() << " detections" << std::endl << std::endl;

    for (auto & detection : detections) {

        // Check first for matching stuff we think is real
        if (matchesAnyFace(detection, definiteFaces, nFrame, true)) {
            continue;

        }
        
        // Check first for matching stuff we think is potential
        if (matchesAnyFace(detection, potentialFaces, nFrame, false)) {
           continue;
        }


        // If we are here, there is no matching face already existing, so 
        // it can be added to list of potentials
        potentialFaces.emplace_back(detection, masks.getNextMask(), nFrame);
        std::cout << "Adding new potential face." << std::endl;
    }

    // Now upgrade those with enough matches
    upgradePotentialFaces();

    // And purge the old ones
    purgeOldFaces(nFrame);

    std::cout << definiteFaces.size() << " real faces." << std::endl;
    std::cout << potentialFaces.size() << " potential faces." << std::endl;

}

// To draw what we have
const std::vector<Face>& FaceTracker::getFaces() const {
    return definiteFaces;
}

// Need to run stats on size and position similarity to check for a match
bool FaceTracker::matchesFace(const cv::Rect & potential, Face & face, const int& nFrame) {
    const cv::Rect& lastSeen = face.lastPosition;
    const int frameDiff = face.undetectedFrames(nFrame);

    // Percent moved in X direction relative to size of object
    double percxdiff = (double)abs(potential.x - lastSeen.x)/lastSeen.width;

    // Percent moved in Y direction relative to size of object
    double percydiff = (double)abs(potential.y - lastSeen.y)/lastSeen.height;

    // Ratio of width change
    double ratiow = (double)potential.width / lastSeen.width;

    // Ration of height change
    double ratioh = (double)potential.height / lastSeen.height;

    std::cout << std::endl;
    std::cout << "Face #" << face.id << " last seen " << frameDiff << " frames ago" << std::endl;
    std::cout << "Percent X: " << percxdiff << " Percent Y: " << percydiff << std::endl;
    std::cout << "Ratio width: " << ratiow << " Ratio height: " << ratioh << std::endl;

    bool match = true;

    // Be more strict if last seen more recently
    if (frameDiff <= 5) {
        if (ratiow < 0.9 || ratiow > 1.11 || ratioh < 0.9 || ratioh > 1.11) {
            std::cout << "No match, width off" << std::endl;
            match = false;
        } else if (percxdiff > 0.75 || percydiff > 0.75) {
            // can't quickly move too far
            std::cout << "No match, moved too far" << std::endl;
            match = false;
        }
    } else {
        if (ratiow < 0.7 || ratiow > 1.43 || ratioh < 0.7 || ratioh > 1.43) {
            std::cout << "No match, width off" << std::endl;
            match = false;
        } else if (percxdiff > 1.5 || percydiff > 1.5) {
            // can't quickly move too far
            std::cout << "No match, moved too far" << std::endl;
            match = false;
        }
    }
    return match;
}

// Check detections against a std::vector of faces for matches, and return
// true if a match is found. 
//
// TODO - for future improvement do a nearest neighbor check rather than return
// on first close match
bool FaceTracker::matchesAnyFace(const cv::Rect& detection, std::vector<Face>& faceVector, const int& nFrame, bool real) {
    bool foundMatch = false;
    for (auto & face : faceVector) {
        if (matchesFace(detection, face, nFrame)) {
            std::cout << "Found a match for a " << (real ? "real" : "potential") << " face!" << std::endl;
            face.updateSeen(detection, nFrame);
            foundMatch = true;
            break;
        }
    }
    return foundMatch;
}

// Purge faces that have not been seen in a while
void FaceTracker::purgeOldFaces(const int& nFrame) {
    definiteFaces.erase(std::remove_if(definiteFaces.begin(), 
                                       definiteFaces.end(), 
                                       [&](Face & i) { return i.undetectedFrames(nFrame) > EXPIRE_FACE_FRAMES; }), 
                        definiteFaces.end());
    potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                       potentialFaces.end(), 
                                       [&](Face & i) { return i.undetectedFrames(nFrame) > EXPIRE_FACE_FRAMES; }), 
                        potentialFaces.end());
}


// Upgrade faces that have met the detection threshold
void FaceTracker::upgradePotentialFaces() {
    // Copy faces over to definiteFaces if enough detections have been met
    std::copy_if(potentialFaces.begin(),
                 potentialFaces.end(),
                 std::back_inserter(definiteFaces), 
                 [](Face & i){return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL;} );
    
    // And remove from potential
    potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                        potentialFaces.end(), 
                                        [](Face & i) { return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL; }), 
                         potentialFaces.end());
}

