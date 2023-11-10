#pragma once
#include "Neuron.h"
class Layer {
public:
	Layer();
	void randWeightsOnConnections(int kol);
	void straightProp(double values[], int kol);
	void printValues();

	void setQuantityOfNeurons(int input);
	int getQuantityOfNeurons();
	void inputDataForWork();
	void setNeuronsValues(double value[]);
	double getNeuronValue(int index);
	
	void say();
private:
	Neuron* layer;
	int quantityOfNeuorns;
};
