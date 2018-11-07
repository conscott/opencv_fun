#include "opencv2/imgproc.hpp"

// Class to store "Face Masks"
// Imagnes with an alpha channel that will be overlayed to mask
// detected faces in the video feed.
class Mask
{

    private:
        // Load from local file in ./imgs
        std::string filename;
        // Need to scale masks differently to fit faces
        double scale;
        // The 4-channel uncompressed image
        cv::Mat img;

    public:

        // Need to provide args, no default constructor
        Mask() = delete;

        Mask(std::string _filename, double _scale) : filename(_filename), scale(_scale) {
            img = cv::imread(filename, cv::IMREAD_UNCHANGED);
        }

        const cv::Mat& getImg() const {
            return img;
        }

        double getScale() const {
            return scale;
        }
};

// Subclass std::vector and add 
class MaskVector : public std::vector<Mask>
{

    private:
        int nextImg = 0;

    public:

        Mask& getNextMask() {
            assert(!this->empty());
            nextImg += 1;
            return this->operator[](nextImg%(this->size()));
        }
};
