#include "Net.h"
#include "fstream"
#include "iostream"
#include <string>
#include <Windows.h>

// Methods for fully-connected layers

using namespace concurrency;

Net::Net(std::string network_path, std::string optimisator, double input_speed, double special_parameters) {
	setParamsFromFile(network_path);
	speed = input_speed;
	specialParams = special_parameters;
	if (optimisator == "sgd") {
		optimisatorFunc = 0;
	}
	if (optimisator == "momentum") {
		optimisatorFunc = 1;
	}


}

std::vector<double> Net::predict(double data[]) {
	layers[0].setNeuronsValues(data);
	straightProp();
	return returnFinals();
}

void Net::randWeights() {
	for (int i = 0; i < quantityOfLayers-1; i++) {
		layers[i].randWeightsOnConnections(layers[i + 1].getQuantityOfNeurons());
	}
	layers[quantityOfLayers - 1].randWeightsOnConnections(0);
}

void Net::readWeights(std::fstream& file) {
	for (int i = 1; i < quantityOfLayers; i++) {
		layers[i].readWeights(layers[i - 1].getQuantityOfNeurons(), file);
	}
}

void Net::setParamsFromFile(std::string network_path) {
	std::fstream file(network_path);
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
		createConnectionsToPrevLayers(false);
	}
	else {
		readWeights(file);
		createConnectionsToNextLayers();
	}
	
	file.close();
}

void Net::train(std::string trainData, double critError, std::string network_save_path) {
	inputDataset(trainData);
	volatile double error;
	int epochs = 0;
	do {
		/*if (epochs % 10 == 0) {
			system("cls");
		}*/
		error = 0;

		for (int c = 0; c < trainDataNumber; c++) {
			layers[0].setNeuronsValues(dataSet[c]);
			straightProp();
			error += squareError(c);
			calcNeuronError(c);
			correctWeights();
			createConnectionsToPrevLayers();
		}
		error /= trainDataNumber;
		epochs++;
		std::cout << error << "\n";
	} while (error > critError);
	std::cout << '\n';
	returnFinals();
	saveData(network_save_path);
}

void Net::straightProp() {
	double* prevValues;
	int neuronsNumber;
	for (int c = 1; c < quantityOfLayers; c++) {
		neuronsNumber = layers[c - 1].getQuantityOfNeurons();
		prevValues = new double[neuronsNumber];
		//std::cout << "1: ";
		for (int i = 0; i < neuronsNumber; i++) {
			prevValues[i] = layers[c - 1].getNeuronValue(i);
			//std::cout << prevValues[i] << ' ';
		}
		//std::cout << '\n';
		layers[c].straightProp(prevValues, neuronsNumber);
		//std::cout << '\n';
		delete[] prevValues;
	}
}

std::vector<double> Net::returnFinals() {
	return layers[quantityOfLayers - 1].returnValues();
}


void Net::inputDataset(std::string data) {
	std::fstream file(data);
	file >> trainDataNumber;
	std::cout << trainDataNumber << std::endl;
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

double Net::squareError(int iterator) {
	double error = 0;
	for (int c = 0; c < layers[quantityOfLayers - 1].getQuantityOfNeurons(); c++) {
		double buffer = (layers[quantityOfLayers - 1].getNeuronValue(c) - dataSet[iterator][layers[0].getQuantityOfNeurons()]);
		error += 0.5 * buffer * buffer;
	}
	return error;
}

void Net::calcNeuronError(int iterator) {
	double error = 0;
	double* errors;
	int len;
	layers[quantityOfLayers - 1].outerNeuronError(dataSet[iterator][layers[0].getQuantityOfNeurons()]);
	for (int i = quantityOfLayers - 2; i > 0; i--) {
		len = layers[i + 1].getQuantityOfNeurons();
		errors = new double[len];
		for (int j = 0; j < len; j++) {
			errors[j] = layers[i+1].getErrors(j);
		}
		layers[i].innerNeuronError(errors, len);
		delete[] errors;
	}
}

void Net::correctWeights() {
	double* errors;
	size_t len;

	for (int c = 0; c < quantityOfLayers - 1; c++) {
		len = layers[c + 1].getQuantityOfNeurons();
		errors = new double[len];
		/*parallel_for(size_t(0), len, [&](size_t j) {
			errors[j] = layers[c + 1].getErrors(j);
			});*/

		for (int j = 0; j < len; j++) {
			errors[j] = layers[c + 1].getErrors(j);
		}

		switch (optimisatorFunc)
		{
		case 0:
			layers[c].correctWeightsSGD(speed, errors);
			break;
		case 1:
			layers[c].correctWeightsMomentum(speed, errors, specialParams);
			break;
		default:
			std::cout << "warning: No such optimisation function.\n";
			break;
		}
		delete[] errors;
	}
}

void Net::createConnectionsToPrevLayers(bool wasAllocated) {
	for (int c = 1; c < quantityOfLayers; c++) {
		double *array = new double [layers[c - 1].getQuantityOfNeurons()];
		for (int i = 0; i < layers[c].getQuantityOfNeurons(); i++) {
			layers[c - 1].getWeightsOfIndexedConnectionsPrev(i, array);
			if(wasAllocated)
				layers[c].createConnectionsToPrev(array, i);
			else {
				layers[c].allocateMemForConnections(layers[c - 1].getQuantityOfNeurons(), i);
				layers[c].createConnectionsToPrev(array, i);
			}
		}
		delete[] array;
	}
}

void Net::createConnectionsToNextLayers() {
	for (int c = 0; c < quantityOfLayers - 1; c++) {
		double* array = new double[layers[c + 1].getQuantityOfNeurons()];
		for (int i = 0; i < layers[c].getQuantityOfNeurons(); i++) {
			layers[c + 1].getWeightsOfIndexedConnectionsNext(i, array);
			layers[c].createConnectionsToNext(array, layers[c + 1].getQuantityOfNeurons(), i);
		}
		delete[] array;
	}
}

void Net::saveData(std::string save_path) {
	std::ofstream file(save_path);
	file << "trained\n" << quantityOfLayers << '\n';
	for (int c = 0; c < quantityOfLayers; c++) {
		file << layers[c].getQuantityOfNeurons() << ' ';
	}
	for (int c = 1; c < quantityOfLayers; c++) {
		layers[c].saveWeights(file);
	}
	file.close();
}


// Methods for convertive layers

Net::Net(std::string network_path, std::string convPath, std::string optimisator, double input_speed, size_t filter_size, double special_parameters) {
	setParamsFromFile(network_path);
	setConvLayersFromFile(convPath);
	speed = input_speed;
	specialParams = special_parameters;
	if (optimisator == "sgd") {
		optimisatorFunc = 0;
	}
	if (optimisator == "momentum") {
		optimisatorFunc = 1;
	}
	nameTraenData.push_back("airplane");
	nameTraenData.push_back("automobile");
	nameTraenData.push_back("bird");
	nameTraenData.push_back("cat");
	nameTraenData.push_back("deer");
	nameTraenData.push_back("dog");
	nameTraenData.push_back("frog");
	nameTraenData.push_back("horse");
	nameTraenData.push_back("ship");
	nameTraenData.push_back("truck");

}

void Net::convRandWeights() {
	for (int c = 0; c < convQuantityOfLayers; c++) {
		convLayers[c].randMatrix(matrixSize);
	}
}

void Net::setConvLayersFromFile(std::string network_path) {
	std::fstream file(network_path);
	std::string state;
	file >> state;
	file >> convQuantityOfLayers >> matrixSize;
	convLayers = new ConvLayer[convQuantityOfLayers];
	int x;
	for (int i = 0; i < convQuantityOfLayers / 2; i++) {
		file >> x;
		convLayers[i].setQuantityOfNeurons(x);
	}

	if (state == "untrained") {
		convRandWeights();
		createConnectionsToPrevLayers(false);
	}
	else {
		readWeights(file);
		createConnectionsToNextLayers();
	}

	file.close();
}

std::string Net::randData() {
	std::string way = "D:\\Tvorch_proect\\cifar10\\train\\";
	int x = rand() % 10;
	int y = rand() % 5000 + 1;
	way += nameTraenData[x] + "\\";
	std::string s = "0000", name = std::to_string(y);
	for (int c = name.size() - 1; c >= 0; c--)
		s[c] = name[c];
	way += s + ".png";
	return way;
}

void Net::say() {
	std::cout << '\n' << convQuantityOfLayers << '\n';
	for (int c = 0; c < convQuantityOfLayers / 2; c++) {
		convLayers[c].say();
	}
}

// Destructor
Net::~Net() {
	//delete[] layers;
	//for (int c = 0; c < trainDataNumber; c++) {
	//	delete[] dataSet[c];
	//}
	//delete[] dataSet;
	//std::cout << "мю мере\n";
}
