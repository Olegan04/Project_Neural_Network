#include "ConvNeuron.h"

void ConvNeuron::fillMatrix(int _size) {
	size = _size;
	RMatrix = new double* [size];
	GMatrix = new double* [size];
	BMatrix = new double* [size];

	RoldDW = new double* [size];
	BoldDW = new double* [size];
	GoldDW = new double* [size];

	for (int i = 0; i < size; i++) {
		RMatrix[i] = new double [size];
		GMatrix[i] = new double [size];
		BMatrix[i] = new double [size];

		RoldDW[i] = new double [size];
		BoldDW[i] = new double [size];
		GoldDW[i] = new double [size];
	}
}

void ConvNeuron::fillResMatrix(int rows, int cols) {
	Rresult = new double* [size];
	Gresult = new double* [size];
	Bresult = new double* [size];
	result = new double* [size];

	for (int i = 0; i < size; i++) {
		Rresult[i] = new double[size];
		Gresult[i] = new double[size];
		Bresult[i] = new double[size];
		result[i] = new double[size];
	}
}

/*void ConvNeuron::forward(double** _red, double** _green, double** _blue, int x, int y) {
	if (result == nullptr)
		fillResMatrix(y, x);

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double red = 0, green = 0, blue = 0;
			for (int k = 0; k < size; k++) {
				for (int q = 0; q < size; q++) {
					red += _red[i + k][j + q] * RMatrix[k][q];
					green += _green[i + k][j + q] * GMatrix[k][q];
					blue += _blue[i + k][j + q] * BMatrix[k][q];
				}
			}
			Rresult[i][j] = red;
			Gresult[i][j] = green;
			Bresult[i][j] = blue;
			result[i][j] = red + green + blue + bias;
		}
	}
}*/

void ConvNeuron::forward(cv::Mat image, int x, int y, Image& uzobr, int neron) {
	int X = 0, Y = 0;
	for (int i = -size / 2; i < y + (size / 2); i++) {
		for (int j = -size / 2; j < x + (size / 2); j++) {
			Y++;
			if (Y == uzobr.size_y) {
				X++;
				Y = 0;
			}
			double red = 0, green = 0, blue = 0;
			for (int k = 0; k < size; k++) {
				for (int q = 0; q < size; q++) {
					if ((i + k) < 0 || (j + q) < 0 || (i + k) >= y || (j + q) >= x) {
						red += 0; green += 0; blue += 0;
					}
					else {
						cv::Vec3b pixel = image.at<cv::Vec3b>(i + k, j + q);
						red += double(pixel[2]) * RMatrix[k][q];
						green += double(pixel[1]) * GMatrix[k][q];
						blue += double(pixel[0]) * BMatrix[k][q];
					}
					
				}
			}
			uzobr.image[X][Y][neron] = red + green + blue + bias;
		}
	}
}