#include "Neuron.h"

void Neuron::sigmoid(double x) {
	value = 1.0 / (1.0 + exp(-x));
}

void Neuron::tanh(double x) {
	value = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

void Neuron::ReLU(double x) {
	value = std::max(0.0, x);
}

void Neuron::leakeReLU(double x) {
	if (x < 0) value = 0.01 * x;
	else value = x;
}