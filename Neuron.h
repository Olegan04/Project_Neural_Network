#pragma once
#include <iostream>
class Neuron {
public:
	double value;
	double error;
	double bias;
	double* connections;
	int quantityOfConnections;

	void sigmoid(double x);
	void tanh(double x);
	void ReLU(double x);
	void leakeReLU(double x);
};