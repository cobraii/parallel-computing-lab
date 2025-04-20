#define _USE_MATH_DEFINES // Добавляем для M_PI
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define A (-M_PI)
#define B M_PI
#define N 1000000

double f(double x) {
    return std::exp(-x * x);
}

double integrate_sequential() {
    double dx = (B - A) / N;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = A + (i + 0.5) * dx;
        sum += f(x);
    }
    return sum * dx;
}

double integrate_parallel() {
    double dx = (B - A) / N;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        double x = A + (i + 0.5) * dx;
        sum += f(x);
    }
    return sum * dx;
}

int main() {
    auto start_seq = std::chrono::high_resolution_clock::now();
    double seq_result = integrate_sequential();
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
    std::cout << "Sequential integral: " << seq_result << "\n";
    std::cout << "Sequential time: " << seq_time << " seconds\n";

    auto start_par = std::chrono::high_resolution_clock::now();
    double par_result = integrate_parallel();
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end_par - start_par).count();
    std::cout << "Parallel integral: " << par_result << "\n";
    std::cout << "Parallel time: " << par_time << " seconds\n";

    std::cout << "Results match: " << (std::abs(seq_result - par_result) < 1e-6 ? "Yes" : "No") << "\n";

    return 0;
}