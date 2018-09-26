#pragma once

#include <stdio.h>
#include <iostream>
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
*/

//OpenCV headers.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void pre_process_image(uchar4 **inputImage, unsigned char **grayImage,
	uchar4 **rgbaImage, unsigned char **d_grayImage,
	const std::string &filename);

void convert_color_to_gray_image();
