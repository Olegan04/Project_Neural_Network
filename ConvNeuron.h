#pragma once
#include <opencv2/opencv.hpp>
#include "Image.h"
class ConvNeuron {
public:
	// ���� ������. 
	// �������� = value, 
	// ������ = error, 
	// �������� = bias, 
	// ��������� ������ � ��������� �������� ��������� �������:
	// ## ��� R-������ = RoldDW,
	// ## ��� G-������ = GoldDW,
	// ## ��� B-������ = BoldDW,
	// ������-�������:
	// ## ��� R-������ = RMatrix,
	// ## ��� G-������ = GMatrix,
	// ## ��� B-������ = BMatrix,
	// ������ 1 ��������� ���������� ������� = size
	double** Rresult = nullptr;
	double** Bresult = nullptr;
	double** Gresult = nullptr;
	double** result = nullptr;
	double error;
	double bias;
	double** RoldDW = nullptr;
	double** BoldDW = nullptr;
	double** GoldDW = nullptr;
	double** RMatrix = nullptr;
	double** GMatrix = nullptr;
	double** BMatrix = nullptr;
	int size;


	void fillMatrix(int _size);
	void fillResMatrix(int rows, int cols);
	void forward(cv::Mat image, int x, int y, Image& output_img, int neron);
};
