#include "Net.h"
#include "fstream"

Net::Net() {
	setParamsFromFile();
}

void Net::randWeights() {
	for (int i = 1; i < quantityOfLayers; i++) {
		layers[i].randWeightsOnConnections(layers[i - 1].getQuantityOfNeurons());
	}
}

void Net::setParamsFromFile() {
	std::fstream file("D:\\Tvorch_proect\\Neural_network\\network_info.txt");
	std::string state;
	file >> state;
	file >> quantityOfLayers;
	layers = new Layer[quantityOfLayers];
	int x;
	for (int i = 0; i < quantityOfLayers; i++) {
		file >> x;
		layers[i].setQuantityOfNeurons(x);
	}

	if (state == "untrained") {
		randWeights();
	}
	else {

	}
	file.close();
}

void Net::train() {
	inputDataset();
	for (int c = 0; c < trainDataNumber; c++) {
		layers[0].setNeuronsValues(dataSet[c]);
		straightProp();
		returnFinals();
	}
}

void Net::straightProp() {
	double* prevValues;
	int neuronsNumber;
	for (int c = 1; c < quantityOfLayers; c++) {
		neuronsNumber = layers[c - 1].getQuantityOfNeurons();
		prevValues = new double[neuronsNumber];

		for (int i = 0; i < neuronsNumber; i++) {
			prevValues[i] = layers[c - 1].getNeuronValue(i);
		}

		layers[c].straightProp(prevValues, neuronsNumber);

		delete prevValues;
	}
}

void Net::returnFinals() {
	layers[quantityOfLayers - 1].printValues();
}

void Net::say() {
	std::cout << '\n' << quantityOfLayers << '\n';
	for (int c = 0; c < quantityOfLayers; c++) {
		layers[c].say();
	}
}

void Net::inputDataset() {
	std::fstream file("D:\\Tvorch_proect\\Neural_network\\dataset.txt");
	file >> trainDataNumber;
	dataSet = new double* [trainDataNumber];
	int n = layers[0].getQuantityOfNeurons();
	for (int i = 0; i < trainDataNumber; i++) {
		dataSet[i] = new double[n + 1];
		for (int c = 0; c <= n; c++) {
			file >> dataSet[i][c];
		}
	}

	file.close();
}