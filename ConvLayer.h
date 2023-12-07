#pragma once
#include "fstream"
#include "vector"
#include "ConvNeuron.h"

class ConvLayer {
public:
	ConvLayer();

	// ����������� ������
	void randMatrix(int size);
	void readMatrix(int size, std::fstream& file);
	void straightProp(double values[], int kol);
	std::vector<double> returnValues();
	void inputDataForWork();
	void outerNeuronError(double y);
	void innerNeuronError(double* errors, int len);
	void correctMatrixSGD(double speed, double* errors);
	void correctMatrixMomentum(double speed, double* errors, double b);


	// �������-�������
	void setQuantityOfNeurons(int input);
	int getQuantityOfNeurons();
	void setNeuronsValues(double value[]);
	double getNeuronValue(int index);
	double getErrors(int iterator);
	void saveMatrix(std::ofstream& file);


	// ������ ������������
	void say();


	~ConvLayer();
private:
	ConvNeuron* layer;
	int quantityOfNeuorns;
};
