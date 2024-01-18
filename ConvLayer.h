#pragma once
#include "fstream"
#include "vector"
#include "ConvNeuron.h"

class ConvLayer {
public:
	ConvLayer();

	// Специальные методы
	//void forward(double** red, double** green, double** blue, int x, int y);
	void forward(cv::Mat image, int x, int y, Image& output_img);
	void randMatrix(int size);
	void readMatrix(int size, std::fstream& file);
	void straightProp(double values[], int kol);
	std::vector<double> returnValues();
	void inputDataForWork();
	void outerNeuronError(double y);
	void innerNeuronError(double* errors, int len);
	void correctMatrixSGD(double speed, double* errors);
	void correctMatrixMomentum(double speed, double* errors, double b);


	// Геттеры-сеттеры
	void setQuantityOfNeurons(int input);
	int getQuantityOfNeurons();
	void setNeuronsValues(double value[]);
	double getNeuronValue(int index);
	double getErrors(int iterator);
	void saveMatrix(std::ofstream& file);


	// Методы тестирования
	void say();

private:
	ConvNeuron* layer = nullptr;
	int quantityOfNeuorns;
};
