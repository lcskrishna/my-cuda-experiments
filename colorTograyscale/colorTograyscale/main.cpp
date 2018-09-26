#include <opencv2/opencv.hpp>
#include <iostream>

#include "process_image.h"

using namespace cv;

Mat imageRGBA;
Mat imageGrey;

int main(int argc, char ** argv)
{
	//Read the image.
	Mat img = imread("lena.png", CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		std::cout << "ERROR: unable to open the file." << std::endl;
		return -1;
	}

	cvtColor(img, imageRGBA, CV_BGR2RGBA);

	//allocate image for the output.
	imageGrey.create(img.rows, img.cols, CV_8UC1);

	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "ERROR: images are not continuous. " << std::endl;
		return -1;
	}




	convert_color_to_gray_image();
	//namedWindow("image", WINDOW_NORMAL);
	//imshow("image", img);
	//waitKey(0);
	return 0;
}