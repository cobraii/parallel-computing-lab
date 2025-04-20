#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

#define SIZE 10000000

void initialize_array(std::vector<double>& arr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    arr.resize(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        arr[i] = dis(gen);
    }
}

double sum_sequential(const std::vector<double>& arr) {
    double sum = 0.0;
    for (int i = 0; i < SIZE; ++i) {
        sum += arr[i];
    }
    return sum;
}

double sum_parallel(const std::vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    std::vector<double> arr;
    initialize_array(arr);

    auto start_seq = std::chrono::high_resolution_clock::now();
    double seq_sum = sum_sequential(arr);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
    std::cout << "Sequential sum: " << seq_sum << "\n";
    std::cout << "Sequential time: " << seq_time << " seconds\n";

    auto start_par = std::chrono::high_resolution_clock::now();
    double par_sum = sum_parallel(arr);
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end_par - start_par).count();
    std::cout << "Parallel sum: " << par_sum << "\n";
    std::cout << "Parallel time: " << par_time << " seconds\n";

    std::cout << "Results match: " << (std::abs(seq_sum - par_sum) < 1e-6 ? "Yes" : "No") << "\n";

    return 0;
}