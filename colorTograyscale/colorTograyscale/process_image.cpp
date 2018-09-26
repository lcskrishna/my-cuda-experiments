#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "process_image.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char * d_grayImage__;

void pre_process_image(uchar4 **inputImage, unsigned char **grayImage,
	uchar4 **rgbaImage, unsigned char **d_grayImage,
	const std::string &filename)
{
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cout << "ERROR: couldn't open the file. " << std::endl;
		exit(1);
	}

}