#include "ConvLayer.h"

ConvLayer::ConvLayer() {

}

int ConvLayer::getQuantityOfNeurons() {
	return quantityOfNeuorns;
}

void ConvLayer::setQuantityOfNeurons(int input) {
	quantityOfNeuorns = input;
	layer = new ConvNeuron[input];
}

void ConvLayer::randMatrix(int size) {
	for (int k = 0; k < quantityOfNeuorns; k++) {

		layer[k].fillMatrix(size);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				layer[k].RMatrix[i][j] = (rand() % 200 - 100) / 100.0;
				layer[k].GMatrix[i][j] = (rand() % 200 - 100) / 100.0;
				layer[k].BMatrix[i][j] = (rand() % 200 - 100) / 100.0;

				layer[k].RoldDW[i][j] = 0;
				layer[k].GoldDW[i][j] = 0;
				layer[k].BoldDW[i][j] = 0;
			}
		}
		layer[k].bias = (rand() % 200 - 100) / 100.0;
	}
}

void ConvLayer::readMatrix(int size, std::fstream& file) {
	for (int i = 0; i < quantityOfNeuorns; i++) {
		layer[i].fillMatrix(size);
		file >> layer[i].bias;

		for (int j = 0; j < quantityOfNeuorns; j++) {
			for (int k = 0; k < size; k++) {
				file >> layer[i].RMatrix[j][k];
				file >> layer[i].GMatrix[j][k];
				file >> layer[i].BMatrix[j][k];
			}
		}
	}
}

void ConvLayer::inputDataForWork() {
	std::fstream file("D:\\Tvorch_proect\\Neural_network\\dataset.txt");
	for (int c = 0; c < quantityOfNeuorns; c++) {
		file >> layer[c].value;
	}

	file.close();
}


void ConvLayer::setNeuronsValues(double value[]) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		layer[c].value = value[c];
	}
}

double ConvLayer::getNeuronValue(int index) {
	return layer[index].value;
}

void ConvLayer::say() {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		for (int i = 0; i < layer[c].size; i++) {
			for (int j = 0; j < layer[c].size; j++) {
				std::cout << layer[c].BMatrix[i][j] << ' ';
			}
			std::cout << '\n';
		}
	}
}
