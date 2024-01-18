#pragma once
#include "Layer.h"
#include "ConvLayer.h"
#include <opencv2/opencv.hpp>
#include "Image.h"

class Net {
public:
	Net(std::string network_path, std::string optimisator, double input_speed = 0.3, double special_parameters = 0.9);
	Net(std::string network_path, std::string convPath, std::string optimisator, double input_speed = 0.3, size_t filter_size = 3, double special_parameters = 0.9);

	// Fully-connected methods
	void inputDataset(std::string data);
	void train(std::string trainData, double critError, std::string network_save_path);

								
	// Testing
	std::vector<double> predict(double data[]);
	std::vector<double> returnFinals();
	void say();


	~Net();

private:
	// Fully-connected vars
	Layer* layers = nullptr;
	int quantityOfLayers; 
	size_t trainDataNumber;
	double** dataSet = nullptr;
	double speed;
	double specialParams;
	int optimisatorFunc;


	// Convertive vars
	size_t matrixSize;
	ConvLayer* convLayers = nullptr;
	int convQuantityOfLayers;
	std::vector <std::string>  nameTraenData;
	double** red = nullptr;
	double** green = nullptr;
	double** blue = nullptr;


	// Fully-connected methods
	void randWeights();
	void readWeights(std::fstream& file);
	void setParamsFromFile(std::string network_path);
	void straightProp();
	double squareError(int iterator);
	void createConnectionsToPrevLayers(bool wasAllocated = true);
	void createConnectionsToNextLayers();
	void calcNeuronError(int iterator);
	void correctWeights();
	void saveData(std::string save_path);


	// Convertive methods
	void train(double critError, std::string network_save_path);
	void convRandWeights();
	void setConvLayersFromFile(std::string network_path);
	std::string randData();
	void cForward(cv::Mat image);
	void fillRGB(int y, int x);
	void maxPull(Image& image, Image& im);
	double max_four(double a, double b, double c, double d);
};