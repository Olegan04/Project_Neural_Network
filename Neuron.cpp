#include "Neuron.h"
#include "algorithm"
#include "iostream"

//Определение методов класса Neuron

void Neuron::sigmoid(double x) {
	value = 1.0 / (1.0 + exp(-x));
	//std::cout << value << ' ';
}

void Neuron::tanh(double x) {
	value = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

void Neuron::ReLU(double x) {
	value = std::max(0.0, x);
}

void Neuron::leakyReLU(double x) {
	if (x < 0) value = 0.01 * x;
	else value = x;
}

double Neuron::sigmoidDerivative() {
	return value * (1 - value);
}

double Neuron::tanhDerivative() {
	return 4 / pow(exp(value) + exp(-value), 2);
}

double Neuron::ReLUDerivative() {
	if (value >= 0) return 1;
	return 0;
}

double Neuron::leakyReLUDerivative() {
	if (value >= 0) return 1;
	return 0.01;
}

Neuron::~Neuron() {
	delete[] connections;
	delete[] connectionsToNextLayer;
	// delete[] oldDW;
}
