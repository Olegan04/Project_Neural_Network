#pragma once
#include <ppl.h>

class Neuron {
public:
	// оНКЪ ЙКЮЯЯЮ. 
	// гмювемхе = value, 
	// ньхайю = error, 
	// ялеыемхе = bias, 
	// беяю ябъгеи я опедшдсыхл якнел = connections, 
	// йнкхвеярбн ябъгеи (дкхмю люяяхбю) = quantityOfConnections,
	// беяю ябъгеи ян якедсчыхл якнел = connectionsToNextLayer,
	// йнкхвеярбн ябъгеи (дкхмю люяяхбю) = int quantityOfConnectionsToNext,
	// хглемемхе ньхайх б опньедьеи хрепюжхх напюрмнцн опнундю = oldDW.
	double value;
	double error;
	double bias;
	double* connections;
	int quantityOfConnections;
	double* connectionsToNextLayer;
	int quantityOfConnectionsToNext;
	double* oldDW;


	// лЕРНДШ ЙКЮЯЯЮ (ТСМЙЖХХ ЮЙРХБЮЖХХ),
	void sigmoid(double x);
	void tanh(double x);
	void ReLU(double x);
	void leakyReLU(double x);
	// (ОПНХГБНДМШЕ ТСМЙЖХИ ЮЙРХБЮЖХХ).
	double sigmoidDerivative();
	double tanhDerivative();
	double ReLUDerivative();
	double leakyReLUDerivative();

	~Neuron();
};