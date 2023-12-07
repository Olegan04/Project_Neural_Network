#pragma once

class Activator {
public:
	// ������� ���������
	double sigmoid(double x);
	double tanh(double x);
	double ReLU(double x);
	double leakyReLU(double x);


	// ����������� ������� ���������
	double sigmoidDerivative(double value);
	double tanhDerivative(double value);
	double ReLUDerivative(double value);
	double leakyReLUDerivative(double value);

};
