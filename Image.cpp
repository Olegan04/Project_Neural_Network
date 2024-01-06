#include "Image.h"
Image::Image(int x, int y, int z) {
	size_x = x;
	size_y = y;
	kol_collor = z;
	image = new double** [size_x];
	for (int i = 0; i < size_x; i++) {
		image[i] = new double* [size_y];
		for (int j = 0; j < size_y; j++) {
			image[i][j] = new double[kol_collor];
		}
	}
}