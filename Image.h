#pragma once
#include <iostream>

class Image {
public:
	int size_x, size_y;
	int kol_collor;
	double*** image;
	Image(int x, int y, int z);
};
