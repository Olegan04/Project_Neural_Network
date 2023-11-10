#include "Layer.h"
#include "algorithm"
#include "fstream"

Layer::Layer() {

}

int Layer::getQuantityOfNeurons() {
	return quantityOfNeuorns;
}

void Layer::setQuantityOfNeurons(int input) {
	quantityOfNeuorns = input;
	layer = new Neuron[input];
}

void Layer::randWeightsOnConnections(int kol) {
	for (int k = 0; k < quantityOfNeuorns; k++) {
		layer[k].connections = new double [kol];
		layer[k].quantityOfConnections = kol;
		for (int c = 0; c < kol; c++) {
			layer[k].connections[c] = (rand() % 200 - 100) / 100.0;
		}
		layer[k].bias = (rand() % 200 - 100) / 100.0;
	}
}

void Layer::inputDataForWork() {
	std::fstream file("D:\\Tvorch_proect\\Neural_network\\dataset.txt");
	for (int c = 0; c < quantityOfNeuorns; c++) {
		file >> layer[c].value;
	}

	file.close();
}

void Layer::setNeuronsValues(double value[]) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		layer[c].value = value[c];
	}
}

double Layer::getNeuronValue(int index) {
	return layer[index].value;
}

void Layer::straightProp(double values[], int kol) {
	int sum;
	for (int c = 0; c < quantityOfNeuorns; c++) {
		sum = layer[c].bias;
		for (int i = 0; i < kol; i++) {
			sum += values[i] * layer[c].connections[i];
		}

		layer[c].sigmoid(sum);
	}
}

void Layer::printValues() {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		std::cout << layer[c].value << "\n";
	}
}

void Layer::say() {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		for (int i = 0; i < layer[c].quantityOfConnections; i++) {
			std::cout << layer[c].connections[i];
		}
	}
}

