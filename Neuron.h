#pragma once
#include <ppl.h>

class Neuron {
public:
	// ���� ������. 
	// �������� = value, 
	// ������ = error, 
	// �������� = bias, 
	// ���� ������ � ���������� ����� = connections, 
	// ���������� ������ (����� �������) = quantityOfConnections,
	// ���� ������ �� ��������� ����� = connectionsToNextLayer,
	// ���������� ������ (����� �������) = int quantityOfConnectionsToNext,
	// ��������� ������ � ��������� �������� ��������� ������� = oldDW.
	double value;
	double error;
	double bias;
	double* connections;
	int quantityOfConnections;
	double* connectionsToNextLayer;
	int quantityOfConnectionsToNext;
	double* oldDW;


	// ������ ������ (������� ���������),
	void sigmoid(double x);
	void tanh(double x);
	void ReLU(double x);
	void leakyReLU(double x);
	// (����������� ������� ���������).
	double sigmoidDerivative();
	double tanhDerivative();
	double ReLUDerivative();
	double leakyReLUDerivative();

	~Neuron();
};