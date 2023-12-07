#pragma once
#include "Neuron.h"
#include <fstream>
#include <vector>

class Layer {
public:
	Layer();

	// Специальные методы
	void randWeightsOnConnections(int kol);
	void readWeights(int kol, std::fstream &file);
	void allocateMemForConnections(int arrayLength, int index);
	void straightProp(double values[], int kol);
	std::vector<double> returnValues();
	void inputDataForWork();
	void createConnectionsToPrev(double *nextToThisConnections, int index);
	void createConnectionsToNext(double* nextToThisConnections, int arrayLength, int index);
	void outerNeuronError(double y);
	void innerNeuronError(double *errors, int len);
	void correctWeightsSGD(double speed, double *errors);
	void correctWeightsMomentum(double speed, double* errors, double b);


	// Геттеры-сеттеры
	void setQuantityOfNeurons(int input);
	int getQuantityOfNeurons();
	void setNeuronsValues(double value[]);
	double getNeuronValue(int index);
	void getWeightsOfIndexedConnectionsPrev(int index, double *array);
	void getWeightsOfIndexedConnectionsNext(int index, double* array);
	double getErrors(int iterator);
	void saveWeights(std::ofstream& file);


	// Методы тестирования
	void say();


	~Layer();
private:
	Neuron* layer;
	int quantityOfNeuorns;
};
