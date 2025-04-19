#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>


std::mutex resultMtx;
long long result = 1;


void calculateFactorial(int start, int end) {
    long long local = 1;

    for (int i = start; i <= end; ++i) {
        local *= i;
    }

    std::lock_guard<std::mutex> lock(resultMtx);
    result *= local;
}


int main() {
    system("chcp 65001");

    int number;
    int threadsCount;

    std::cout << "Введите число для вычисления факториала: ";
    std::cin >> number;

    std::cout << "Введите количество потоков: ";
    std::cin >> threadsCount;

    std::vector<std::thread> threads;

    if (number < 1 || threadsCount < 1) {
        std::cout << "Неверный формат данных";
        return 1;
    }

    int chunkSize = number / threadsCount;
    int start = 1;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < threadsCount; ++i) {
        int end = (i == threadsCount - 1) ? number : start + chunkSize - 1;
        threads.emplace_back(calculateFactorial, start, end);
        start = end + 1;
    }

    for (auto &thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

    std::cout << "Факториал числа " << number << " равен " << result << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " нс." << std::endl;

    return 0;
}