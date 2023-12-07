#include "Layer.h"
#include "algorithm"
#include "fstream"
#include "iostream"

//Определение методов класса Layer

using namespace concurrency;

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
		layer[k].connectionsToNextLayer = new double [kol];
		layer[k].oldDW = new double [kol];
		layer[k].quantityOfConnectionsToNext = kol;
		for (int c = 0; c < kol; c++) {
			layer[k].connectionsToNextLayer[c] = (rand() % 200 - 100) / 100.0;
			layer[k].oldDW[c] = 0;
			//std::cout << layer[k].connectionsToNextLayer[c] << ' ';
		}
		//std::cout << '\n';
		layer[k].bias = (rand() % 200 - 100) / 100.0;
	}
}

void Layer::readWeights(int kol, std::fstream& file) {
	for (int k = 0; k < quantityOfNeuorns; k++) {
		layer[k].connections = new double[kol];
		layer[k].quantityOfConnections = kol;
		file >> layer[k].bias;
		for (int c = 0; c < kol; c++) {
			file >> layer[k].connections[c];
		}
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
	double sum;
	for (int c = 0; c < quantityOfNeuorns; c++) {
		sum = layer[c].bias;
		for (int i = 0; i < kol; i++) {
			sum += values[i] * layer[c].connections[i];
		}
		layer[c].tanh(sum);
	}
}

void Layer::outerNeuronError(double y) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		layer[c].error = layer[c].tanhDerivative() * (layer[c].value - y);
	}
}

void Layer::innerNeuronError(double *errors, int len) {
	double sum;
	for (int c = 0; c < quantityOfNeuorns; c++) {
		sum = 0;
		for (int i = 0; i < len; i++) {
			sum += layer[c].connectionsToNextLayer[i] * errors[i];
		}
		layer[c].error = layer[c].tanhDerivative() * sum;
	}
}

void Layer::correctWeightsSGD(double speed, double *errors) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		layer[c].bias -= speed * layer[c].error;
		for (int i = 0; i < layer[c].quantityOfConnectionsToNext; i++) {
			layer[c].connectionsToNextLayer[i] -= speed * errors[i] * layer[c].value;
		}
	}
}

void Layer::correctWeightsMomentum(double speed, double* errors, double b) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		layer[c].bias -= speed * layer[c].error;
		/*size_t numberOfConnections = layer[c].quantityOfConnectionsToNext;
		parallel_for (size_t(0), numberOfConnections, [&](size_t i) {
			layer[c].oldDW[i] = b * layer[c].oldDW[i] + (1 - b) * errors[i] * layer[c].value;
			layer[c].connectionsToNextLayer[i] -= speed * layer[c].oldDW[i];
		});*/

		for (int i = 0; i < layer[c].quantityOfConnectionsToNext; i++) {
			layer[c].oldDW[i] = b * layer[c].oldDW[i] + (1 - b) * errors[i] * layer[c].value;
			layer[c].connectionsToNextLayer[i] -= speed * layer[c].oldDW[i];
		}
	}
}

std::vector<double> Layer::returnValues() {
	std::vector<double> results;
	for (int c = 0; c < quantityOfNeuorns; c++) {
		results.push_back(layer[c].value);
	}
	//std::cout << results.size();
	return results;
}

void Layer::getWeightsOfIndexedConnectionsPrev(int index, double* array) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		array[c] = layer[c].connectionsToNextLayer[index];
	}
}

void Layer::getWeightsOfIndexedConnectionsNext(int index, double* array) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		array[c] = layer[c].connections[index];
	}
}

void Layer::allocateMemForConnections(int arrayLength, int index) {
	layer[index].connections = new double[arrayLength];
	layer[index].quantityOfConnections = arrayLength;
}

void Layer::createConnectionsToPrev(double* nextToThisConnections, int index) {
	for (int i = 0; i < layer[index].quantityOfConnections; i++) {
		layer[index].connections[i] = nextToThisConnections[i];
	}
}

void Layer::createConnectionsToNext(double* nextToThisConnections, int arrayLength, int index) {
	layer[index].connectionsToNextLayer = new double[arrayLength];
	layer[index].oldDW = new double[arrayLength];
	layer[index].quantityOfConnectionsToNext = arrayLength;
	for (int i = 0; i < arrayLength; i++) {
		layer[index].connectionsToNextLayer[i] = nextToThisConnections[i];
		layer[index].oldDW[i] = 0;
	}
}

double Layer::getErrors(int iterator) {
	return layer[iterator].error;
}

void Layer::saveWeights(std::ofstream& file) {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		file << '\n' << layer[c].bias;
		for (int i = 0; i < layer[c].quantityOfConnections; i++) {
			file << ' ' << layer[c].connections[i];
 		}
	}
}

void Layer::say() {
	for (int c = 0; c < quantityOfNeuorns; c++) {
		for (int i = 0; i < layer[c].quantityOfConnectionsToNext; i++) {
			std::cout << layer[c].connectionsToNextLayer[i] << " ЖОПА\n";
		}
	}
}

Layer::~Layer() {
	delete[] layer;
	/*std::cout << "НА ЛЕЕРЕ\n";*/
}
