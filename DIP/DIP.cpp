// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <tuple>
#include<string>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "backprop.h"


enum Shapes {
	Square,
	Star,
	Rectangle,
	Circle,

	None
};


class Object {

public:

	int m00;
	int m10;
	int m01;

	int permi;
	int xt;
	int yt;

	int u11;
	int u02;
	int u20;

	long double umax;
	long double umin;

	double F1;
	double F2;
	
	Shapes Class;

	Object() {
		m00 = m10 = m01 = permi = xt = yt = u11 = u02 = u20 = umax = umin = F1 = F2 = 0;
		Class = Shapes::None;
	}


	std::string getClassType() {


		switch (Class)
		{
		case Square:
			return "Square";
			break;
		case Rectangle:
			return "Rectangle";

			break;
		case Star:
			return "Star";
			break;

		case Circle:
			return "Circle";
			break;

		case None:
		default:
			return "None";
			break;
		}

	}


	void PrintValues() {
		std::cout << "Area: " << m00 << " m10: " << m10 << " m01: " << m01 << " xt: " << xt << " yt: " << yt << " Permi: " << permi;
		std::cout << " u02: " << u02 << " u20: " << u20 << " u11: " << u11;
		std::cout << " F1:" << F1 << " F2: " << F2 << std::endl;


	}

};

bool floodFill(int y, int x,int& index, bool ** visited, cv::Mat& shapes_8uc1_img,
	cv::Mat& indexed_8uc1_img,int& rows,int &columns,
	Object & Shape
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

	Shape.m00 += pow(x, 0) * pow(y, 0);
	Shape.m10 += pow(x, 1) * pow(y, 0);
	Shape.m01 += pow(x, 0) * pow(y, 1);

	bool contribute = false;

	contribute |=floodFill(y + 1, x, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows,columns,Shape);
	contribute |= floodFill(y - 1, x, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);
	contribute |= floodFill(y, x + 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns, Shape);
	contribute |= floodFill(y, x - 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);

	floodFill(y + 1, x+1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);
	floodFill(y + 1, x-1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);
	floodFill(y -1, x + 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);
	floodFill(y -1, x - 1, index, visited, shapes_8uc1_img, indexed_8uc1_img, rows, columns,Shape);


	if (contribute) {
		Shape.permi++;
	}

	return false;
	

}

void floodFillFeatures(int y, int x, bool** visited, cv::Mat& shapes_8uc1_img,
	int& rows, int& columns,Object& Shape
) {


	if (y < 0 || y == rows || x < 0 || x == columns)
		return;

	if (visited[y][x]) {
		return;
	}

	if (shapes_8uc1_img.at<uchar>(y, x) == 0) {
		return;
	}


	visited[y][x] = true;

	Shape.u20 += pow(x - Shape.xt, 2) * pow(y - Shape.yt, 0);
	Shape.u02 += pow(x - Shape.xt, 0) * pow(y - Shape.yt, 2);
	Shape.u11 += pow(x - Shape.xt, 1) * pow(y - Shape.yt, 1);


	floodFillFeatures(y + 1, x,  visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y - 1, x,  visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y, x + 1,  visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y, x - 1, visited, shapes_8uc1_img, rows, columns,Shape);

	floodFillFeatures(y + 1, x + 1, visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y + 1, x - 1,  visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y - 1, x + 1,  visited, shapes_8uc1_img, rows, columns,Shape);
	floodFillFeatures(y - 1, x - 1,  visited, shapes_8uc1_img, rows, columns,Shape);


}


std::vector<Object> calculateFeatures(const char * image , cv::Mat & src_8uc3_img, cv::Mat & indexed_8uc1_img_rt , cv::Mat & threshold_8uc1_img_rt) {
	std::vector<Object> Shapes;

	src_8uc3_img = cv::imread(image, cv::IMREAD_GRAYSCALE);

	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return Shapes;
	}

	int rows = src_8uc3_img.rows, columns = src_8uc3_img.cols;

	cv::Mat threshold_8uc1_img(rows, columns, CV_8UC1);
	cv::Mat indexed_8uc1_img(rows, columns, CV_8UC1);

	bool** visited = (bool**)calloc(rows, sizeof(bool*));
	bool** visited2 = (bool**)calloc(rows, sizeof(bool*));



	for (int i = 0; i < rows; ++i) {
		visited[i] = (bool*)calloc(columns, sizeof(bool));
		visited2[i] = (bool*)calloc(columns, sizeof(bool));

	}

	int threshold = 128;

	for (int y = 0; y < src_8uc3_img.rows; y++) {
		for (int x = 0; x < src_8uc3_img.cols; x++) {

			if (src_8uc3_img.at<uchar>(y, x) > threshold) {

				threshold_8uc1_img.at<uchar>(y, x) = 255;
			}
			else {
				threshold_8uc1_img.at<uchar>(y, x) = 0;

			}
			indexed_8uc1_img.at<uchar>(y, x) = 0;
		}
	}

	int index = 20;

	for (int y = 0; y < threshold_8uc1_img.rows; y++) {
		for (int x = 0; x < threshold_8uc1_img.cols; x++) {

			if (!visited[y][x] && threshold_8uc1_img.at<uchar>(y, x) == 255) {

				Object Shape{};

				floodFill(y, x, index, visited, threshold_8uc1_img, indexed_8uc1_img, rows, columns, Shape);
				index += 15;

				Shape.xt = Shape.m10 / Shape.m00;
				Shape.yt = Shape.m01 / Shape.m00;


				Shape.F1 = Shape.permi * Shape.permi / (100.0 * Shape.m00);

				floodFillFeatures(y, x, visited2, threshold_8uc1_img, rows, columns, Shape);

				Shape.umax = 0.5 * (Shape.u20 + Shape.u02) + 0.5 * sqrtl(4ll * powl(Shape.u11, 2) + powl(Shape.u20 - Shape.u02, 2));
				Shape.umin = 0.5 * (Shape.u20 + Shape.u02) - 0.5 * sqrtl(4ll * powl(Shape.u11, 2) + powl(Shape.u20 - Shape.u02, 2));

				Shape.F2 = Shape.umin / Shape.umax;

				Shapes.push_back(Shape);


			}
		}
	}

	indexed_8uc1_img_rt = indexed_8uc1_img;
	threshold_8uc1_img_rt = threshold_8uc1_img;

	return Shapes;

}

void train(NN* nn , std::vector<Object> trainSet)
{
	int n = trainSet.size();
	double** trainingSet = new double* [n];

	for (int i = 0; i < trainSet.size(); i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];


		trainingSet[i][0] = trainSet[i].F1;
		trainingSet[i][1] = trainSet[i].F2;

		switch (trainSet[i].Class)
		{
		case Shapes::Square:
			trainingSet[i][nn->n[0]] = 1.0;
			trainingSet[i][nn->n[0]+1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
			trainingSet[i][nn->n[0] + 3] = 0.0;

			break;

		case Shapes::Rectangle:
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 1.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
			trainingSet[i][nn->n[0] + 3] = 0.0;

			break;

		case Shapes::Star:
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 1.0;
			trainingSet[i][nn->n[0] + 3] = 0.0;



			break;

		case Shapes::Circle:
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
			trainingSet[i][nn->n[0] + 3] = 1.0;



			break;

		default:
			break;
		}

	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i % n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		//printf( "\rerr=%0.3f", error );
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, std::vector<Object> & testSet , cv::Mat & threshold_8uc1_img_2, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	double font_size = 0.5;
	cv::Scalar font_Color(128, 128, 128);

	for (int n = 0; n < testSet.size(); n++) {

		in[0] = testSet[n].F1;
		in[1] = testSet[n].F2;

		setInput(nn, in, true);
		feedforward(nn);

		int output = getOutput(nn, true);


		switch (output)
		{
		case 0:
			testSet[n].Class = Shapes::Square;
			break;
		case 1:
			testSet[n].Class = Shapes::Rectangle;
			break;
		case 2:
			testSet[n].Class = Shapes::Star;
			break;
		case 3:
			testSet[n].Class = Shapes::Circle;
			break;

		default:
			break;
		}

		cv::Point text_position(testSet[n].xt, testSet[n].yt);
		std::string s = testSet[n].getClassType();

		putText(threshold_8uc1_img_2, s, text_position, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, font_size, font_Color, 1);

	}

}






int main()
{

	cv::Mat src_8uc3_img, indexed_8uc1_img, threshold_8uc1_img;
	std::vector<Object> Shapes = calculateFeatures("images/trainAll.png", src_8uc3_img, indexed_8uc1_img, threshold_8uc1_img);

	int idx = 0;

	for (int i = 1; i <= Shapes.size(); i++) {
		Shapes[i - 1].Class = (::Shapes)idx;

		if (i % 4 ==0) {
			idx++;
		}
	}

	NN* nn = createNN(2, 4, 4);
	train(nn, Shapes);



	cv::Mat src_8uc3_img_2, indexed_8uc1_img_2, threshold_8uc1_img_2;
	std::vector<Object> Shapes2 = calculateFeatures("images/test06.png", src_8uc3_img_2, indexed_8uc1_img_2, threshold_8uc1_img_2);


	test(nn ,Shapes2, threshold_8uc1_img_2, 1000);

	/////////////////////////////////

	cv::imshow("TrainOriginal", src_8uc3_img);
	cv::imshow("TrainThreshold", threshold_8uc1_img);
	cv::imshow("TrainIndexed", indexed_8uc1_img);


	/////////////////////////////////

	cv::imshow("TestOriginal", src_8uc3_img_2);
	cv::imshow("TestThreshold", threshold_8uc1_img_2);
	cv::imshow("TestIndexed", indexed_8uc1_img_2);

	cv::waitKey(0);

	releaseNN(nn);

	return 0;
}