#pragma once
#include "Layer.h";
class Net {
public:
	Net();
	void setParamsFromFile();
	void inputDataset();
	void say();
	void straightProp();
	void returnFinals();
	void train();
private:
	Layer* layers;
	int quantityOfLayers; 
	int trainDataNumber;
	double** dataSet;

	void randWeights();
};