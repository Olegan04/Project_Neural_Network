#include "ConvNeuron.h"

void ConvNeuron::fillMatrix(int _size) {
	size = _size;
	RMatrix = new double* [size];
	GMatrix = new double* [size];
	BMatrix = new double* [size];

	RoldDW = new double* [size];
	BoldDW = new double* [size];
	GoldDW = new double* [size];

	for (int i = 0; i < size; i++) {
		RMatrix[i] = new double [size];
		GMatrix[i] = new double [size];
		BMatrix[i] = new double [size];

		RoldDW[i] = new double [size];
		BoldDW[i] = new double [size];
		GoldDW[i] = new double [size];
	}
}