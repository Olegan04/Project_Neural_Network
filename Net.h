#pragma once
#include "Layer.h"

class Net {
public:
	Net(std::string network_path, std::string optimisator, double input_speed = 0.3, double special_parameters = 0.9);
	Net(std::string network_path, std::string optimisator, double input_speed = 0.3, size_t filter_size = 3, double special_parameters = 0.9);

	// Специальные методы для нейронной регрессии
	void inputDataset(std::string data);
	void train(std::string trainData, double critError, std::string network_save_path);
	std::vector<double> predict(double data[]);
	std::vector<double> returnFinals();


	// Специальные методы для распознавания изображений
	

								
	// Методы тестирования
	void say();


	~Net();

private:
	Layer* layers;
	int quantityOfLayers; 
	size_t trainDataNumber;
	double** dataSet;
	double speed;
	double specialParams;
	int optimisatorFunc;

	void randWeights();
	void readWeights(std::fstream& file);
	void setParamsFromFile(std::string network_path);
	void setFromFileConv(std::string network_path);
	void straightProp();
	double squareError(int iterator);
	void createConnectionsToPrevLayers(bool wasAllocated = true);
	void createConnectionsToNextLayers();
	void calcNeuronError(int iterator);
	void correctWeights();
	void saveData(std::string save_path);
};