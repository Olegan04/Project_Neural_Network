#include <iostream>
#include <ctime>
#include <fstream>
#include "Net.h"
#include <opencv2/opencv.hpp>

int main()
{
    //// Read the image file 
    //cv::Mat image = cv::imread("C:\\Users\\Oleg\\Pictures\\red.png");
    //// Check for failure 
    //if (image.empty())
    //{
    //    std::cout << "Image Not Found!!!" << '\n';
    //    std::cin.get(); //wait for any key press 
    //    return -1;
    //}
    //cv::Vec3b pixel = image.at<cv::Vec3b>(image.rows / 2, image.cols / 2);
    //std::cout << "Значения RGB пикселя в центре изображения: " << "R=" << (int)pixel[2] << ", G=" << (int)pixel[1] << ", B=" << (int)pixel[0] << '\n';


    ////system("cls");

    //// Wait for any keystroke in the window 
    //cv::waitKey(0);
    //return 0;

    //unsigned int start_time = clock();
    setlocale(LC_ALL, "Rus");
    Net network("network_info.txt", "conv_info.txt", "momentum", 0.3, 3, 0.9);
    network.say();

    /*network.train("D:\\Tvorch_proect\\Neural_network\\test.txt", 0.0000132, "network_info_test.txt");
    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    std::cout << '\n' << search_time / 1000.0;*/


    /*double data[] = { 0, 1 };
    auto vector = network.predict(data);
    std::cout << vector[0] << '\n';*/

    return 0;
}
