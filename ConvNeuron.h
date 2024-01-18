#pragma once
#include <opencv2/opencv.hpp>
#include "Image.h"
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
	double** Rresult = nullptr;
	double** Bresult = nullptr;
	double** Gresult = nullptr;
	double** result = nullptr;
	double error;
	double bias;
	double** RoldDW = nullptr;
	double** BoldDW = nullptr;
	double** GoldDW = nullptr;
	double** RMatrix = nullptr;
	double** GMatrix = nullptr;
	double** BMatrix = nullptr;
	int size;


	void fillMatrix(int _size);
	void fillResMatrix(int rows, int cols);
	void forward(cv::Mat image, int x, int y, Image& output_img, int neron);
};
