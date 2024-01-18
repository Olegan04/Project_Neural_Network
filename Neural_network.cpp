#include "Net.h"

int main()
{
    setlocale(LC_ALL, "Rus");
    Net network("network_info.txt", "conv_info.txt", "momentum",  0.3, 3, 0.9);
    //network.train("D:\\Tvorch_proect\\Neural_network\\test.txt", 0.00001, "network_info_test.txt");
    network.say();

    /*double data[3];
    while (true) {
        std::cout << "Введите входные значения сети\n";
        std::cin >> data[0] >> data[1] >> data[2];

       auto vector = network.predict(data);
       std::cout << "Выходное значение сети:\n";
       std::cout << round(vector[0]) << '\n';
    }*/
    

    return 0;
}
