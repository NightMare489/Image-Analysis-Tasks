#include "stdafx.h"
#include <stdio.h>

bool visited[158][264]{};
cv::Mat shapes_8uc1_img(158, 264, CV_8UC1);

int index = 0;

void floodFill(int y, int x) {

	if (visited[y][x]) {
		return;
	}

	if (y < 0 || y == 158 || x < 0 || x == 264)
		return;

	if (shapes_8uc1_img.at<uchar>(y, x) == 0) {
		return;
	}


	visited[y][x] = true;

	//std::cout << index << std::endl;

	shapes_8uc1_img.at<uchar>(y, x) = index;

	floodFill(y + 1, x);
	floodFill(y - 1, x);
	floodFill(y, x + 1);
	floodFill(y, x - 1);

}


int main()
{
	cv::Mat src_8uc3_img = cv::imread("images/train.png", cv::COLOR_BGR2GRAY); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	int threshold = 128;

	for (int y = 0; y < src_8uc3_img.rows; y++) {
		for (int x = 0; x < src_8uc3_img.cols; x++) {

			if (src_8uc3_img.at<uchar>(y, x) > threshold) {

				shapes_8uc1_img.at<uchar>(y, x) = 255;
			}
			else {
				shapes_8uc1_img.at<uchar>(y, x) = 0;

			}

		}
	}



	for (int y = 0; y < shapes_8uc1_img.rows; y++) {
		for (int x = 0; x < shapes_8uc1_img.cols; x++) {

			if (!visited[y][x] && shapes_8uc1_img.at<uchar>(y, x) == 255) {
				floodFill(y, x);
				index += 10;

			}
		}
	}


	cv::imshow("Original", src_8uc3_img);
	cv::imshow("Threshold", shapes_8uc1_img);
	cv::waitKey(0);

}