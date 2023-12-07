#pragma once
#include <iostream>

class ConvNeuron {
public:
	// ���� ������. 
	// �������� = value, 
	// ������ = error, 
	// �������� = bias, 
	// ��������� ������ � ��������� �������� ��������� �������:
	// ## ��� R-������ = RoldDW,
	// ## ��� G-������ = GoldDW,
	// ## ��� B-������ = BoldDW,
	// ������-�������:
	// ## ��� R-������ = RMatrix,
	// ## ��� G-������ = GMatrix,
	// ## ��� B-������ = BMatrix,
	// ������ 1 ��������� ���������� ������� = size
	double value;
	double error;
	double bias;
	double** RoldDW;
	double** BoldDW;
	double** GoldDW;
	double** RMatrix;
	double** GMatrix;
	double** BMatrix;
	int size;


	void fillMatrix(int _size);
};
