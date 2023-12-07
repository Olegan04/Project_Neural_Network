#include "Activator.h"
#include "algorithm"

double Activator::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double Activator::tanh(double x) {
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double Activator::ReLU(double x) {
	return std::max(0.0, x);
}

double Activator::leakyReLU(double x) {
	if (x < 0) return 0.01 * x;
	return x;
}

double Activator::sigmoidDerivative(double value) {
	return value * (1 - value);
}

double Activator::tanhDerivative(double value) {
	return 4 / pow(exp(value) + exp(-value), 2);
}

double Activator::ReLUDerivative(double value) {
	if (value >= 0) return 1;
	return 0;
}

double Activator::leakyReLUDerivative(double value) {
	if (value >= 0) return 1;
	return 0.01;
}