#include "stdafx.h"
#include <vector>
using namespace std;

struct Ck {

	int R, G, B, x, y;

};


int calcGrad(int x, int y, cv::Mat& image) {

	if (x + 1 >= image.cols) {
		x = image.cols - 2;

	}

	if (y + 1 >= image.rows) {
		y = image.rows - 2;
	}

	return
		image.at<cv::Vec3b>(y + 1, x + 1)[0] - image.at<cv::Vec3b>(y, x)[0]
		+ image.at<cv::Vec3b>(y + 1, x + 1)[1] - image.at<cv::Vec3b>(y, x)[1]
		+ image.at<cv::Vec3b>(y + 1, x + 1)[2] - image.at<cv::Vec3b>(y, x)[2];
}


int main() {

	cv::Mat image = cv::imread("images/slic_bears.jpg", cv::IMREAD_COLOR);

	cv::cvtColor(image, image, cv::COLOR_BGR2Lab);

	double font_size = 1;
	const int k = 15;
	int n = image.cols * image.rows;
	int S = sqrt(n / k);

	vector<Ck> segmentCenter;

	cv::Scalar font_Color(0, 0, 255);

	for (int i = S / 2; i < image.cols; i += S) {
		for (int j = S / 2; j < image.rows; j += S) {

			Ck center;

			center.x = i;
			center.y = j;

			center.R = image.at<cv::Vec3b>(j, i)[2];
			center.G = image.at<cv::Vec3b>(j, i)[1];
			center.B = image.at<cv::Vec3b>(j, i)[0];

			segmentCenter.push_back(center);

		}
	}

	for (int i = 0; i < segmentCenter.size(); i++) {


		int clusterGrad = calcGrad(segmentCenter[i].x, segmentCenter[i].y, image);

		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {

				int newGrad = calcGrad(segmentCenter[i].x + dx, segmentCenter[i].y + dy, image);

				if (newGrad < clusterGrad) {

					segmentCenter[i].x += dx;
					segmentCenter[i].y += dy;


					segmentCenter[i].R = image.at<cv::Vec3b>(segmentCenter[i].y, segmentCenter[i].x)[2];
					segmentCenter[i].G = image.at<cv::Vec3b>(segmentCenter[i].y, segmentCenter[i].x)[1];
					segmentCenter[i].B = image.at<cv::Vec3b>(segmentCenter[i].y, segmentCenter[i].x)[0];


				}
			}
		}

	}



	vector<vector<pair<int, int>>> clusters(segmentCenter.size());
	vector < vector<pair<double, int>>> distances(image.cols, vector<pair<double, int>>(image.rows, { DBL_MAX,0 }));

	for (int zz = 0; zz < 10; zz++) {
		clusters = vector<vector<pair<int, int>>>(segmentCenter.size());
		for (int k = 0; k < segmentCenter.size(); k++) {

			for (int x = max(0, segmentCenter[k].x - 2 * S); x < min(segmentCenter[k].x + 2 * S, image.cols); x++) {

				for (int y = max(0, segmentCenter[k].y - 2 * S); y < min(segmentCenter[k].y + 2 * S, image.rows); y++) {

					int R = image.at<cv::Vec3b>(y, x)[2];
					int G = image.at<cv::Vec3b>(y, x)[1];
					int B = image.at<cv::Vec3b>(y, x)[0];


					double Drgb = sqrt(pow(R - segmentCenter[k].R, 2) + pow(G - segmentCenter[k].G, 2) + pow(B - segmentCenter[k].B, 2));

					double Dxy = sqrt(pow(x - segmentCenter[k].x, 2) + pow(y - segmentCenter[k].y, 2));

					const double m = 21;

					double Ds = Drgb + (m / S) * Dxy;

					if (Ds < distances[x][y].first) {
						distances[x][y].first = Ds;
						distances[x][y].second = k;

					}

				}
			}
		}


		for (int i = 0; i < distances.size(); i++) {
			for (int j = 0; j < distances[i].size(); j++) {
				clusters[distances[i][j].second].push_back({ i,j });

			}
		}


		for (int i = 0; i < clusters.size(); i++) {

			int sumx = 0;
			int sumy = 0;


			for (int j = 0; j < clusters[i].size(); j++) {

				sumx += clusters[i][j].first;
				sumy += clusters[i][j].second;

			}

			sumx /= clusters[i].size();
			sumy /= clusters[i].size();


			segmentCenter[i].x = sumx;
			segmentCenter[i].y = sumy;

			segmentCenter[i].R = image.at<cv::Vec3b>(sumy, sumx)[2];
			segmentCenter[i].G = image.at<cv::Vec3b>(sumy, sumx)[1];
			segmentCenter[i].B = image.at<cv::Vec3b>(sumy, sumx)[0];



		}


	}


	cv::Mat image2(image.rows, image.cols, 1124024336);

	for (int i = 0; i < clusters.size(); i++) {
		for (int j = 0; j < clusters[i].size(); j++) {

			image2.at<cv::Vec3b>(clusters[i][j].second, clusters[i][j].first)[2] = segmentCenter[i].R;
			image2.at<cv::Vec3b>(clusters[i][j].second, clusters[i][j].first)[1] = segmentCenter[i].G;
			image2.at<cv::Vec3b>(clusters[i][j].second, clusters[i][j].first)[0] = segmentCenter[i].B;

		}
	}


	cv::cvtColor(image, image, cv::COLOR_Lab2BGR);
	cv::cvtColor(image2, image2, cv::COLOR_Lab2BGR);


	for (int i = 0; i < segmentCenter.size(); i++) {

		Ck superpixel = segmentCenter[i];

		cv::Point text_position(superpixel.x, superpixel.y);
		std::string s = ".";

		putText(image, s, text_position, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, font_size, font_Color, 2);

	}


	cv::imshow("RGB", image2);
	cv::imshow("Original", image);
	cv::waitKey(0);

	return 0;
}
