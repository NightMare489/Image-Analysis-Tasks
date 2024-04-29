#include "stdafx.h"
#include <stdio.h>

int zz = 0;

bool floodFill(int y, int x, int& index, bool** visited, cv::Mat& shapes_8uc1_img,
	cv::Mat& indexed_8uc1_img, int& rows, int& columns,
	int& m00, int& m10, int& m01, int& permi
) {


	if (y < 0 || y == rows || x < 0 || x == columns)
		return true;

	if (visited[y][x]) {
		return false;
	}

	if (shapes_8uc1_img.at<uchar>(y, x) == 0) {
		return true;
	}


	visited[y][x] = true;

	indexed_8uc1_img.at<uchar>(y, x) = index;

	m00 += pow(x, 0) * pow(y, 0);
	m10 += pow(x, 1) * pow(y, 0);
	m01 += pow(x, 0) * pow(y, 1);

	bool contribute = false;

	contribute |= floodFill(y + 1, x, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	contribute |= floodFill(y - 1, x, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	contribute |= floodFill(y, x + 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	contribute |= floodFill(y, x - 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);

	floodFill(y + 1, x + 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	floodFill(y + 1, x - 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	floodFill(y - 1, x + 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
	floodFill(y - 1, x - 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);


	if (contribute) {
		permi++;
	}

	return false;


}


int main()
{
	cv::Mat src_8uc3_img = cv::imread("images/train.png", cv::COLOR_BGR2GRAY); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return -1;
	}

	int rows = src_8uc3_img.rows, columns = src_8uc3_img.cols;

	cv::Mat shapes_8uc1_img(rows, columns, CV_8UC1);
	cv::Mat indexed_8uc1_img(rows, columns, CV_8UC1);

	bool** visited = (bool**)calloc(rows, sizeof(bool*));

	for (int i = 0; i < rows; ++i) {
		visited[i] = (bool*)calloc(columns, sizeof(bool));
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
			indexed_8uc1_img.at<uchar>(y, x) = 0;
		}
	}

	int index = 20;

	for (int y = 0; y < shapes_8uc1_img.rows; y++) {
		for (int x = 0; x < shapes_8uc1_img.cols; x++) {

			if (!visited[y][x] && shapes_8uc1_img.at<uchar>(y, x) == 255) {

				int m00 = 0;
				int m10 = 0;
				int m01 = 0;
				int permi = 0;
				floodFill(y, x, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, m00, m10, m01, permi);
				index += 15;

				int xt = m10 / m00;
				int yt = m01 / m00;
				std::cout << "Area: " << m00 << " m10: " << m10 << " m01: " << m01 << " xt: " << xt << " yt: " << yt << " Permi: " << permi << std::endl;
				zz = 0;

			}


		}
	}


	cv::imshow("Original", src_8uc3_img);
	cv::imshow("Threshold", shapes_8uc1_img);
	cv::imshow("Indexed", indexed_8uc1_img);




	cv::waitKey(0); // wait until keypressed

	return 0;
}