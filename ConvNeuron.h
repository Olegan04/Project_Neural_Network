#pragma once
#include <iostream>

class ConvNeuron {
public:
	// оНКЪ ЙКЮЯЯЮ. 
	// гмювемхе = value, 
	// ньхайю = error, 
	// ялеыемхе = bias, 
	// хглемемхе ньхайх б опньедьеи хрепюжхх напюрмнцн опнундю:
	// ## дкъ R-йюмюкю = RoldDW,
	// ## дкъ G-йюмюкю = GoldDW,
	// ## дкъ B-йюмюкю = BoldDW,
	// тхкэрп-люрпхжю:
	// ## дкъ R-йюмюкю = RMatrix,
	// ## дкъ G-йюмюкю = GMatrix,
	// ## дкъ B-йюмюкю = BMatrix,
	// пюглеп 1 хглепемхъ йбюдпюрмни люрпхжш = size
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
