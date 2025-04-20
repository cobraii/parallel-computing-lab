#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

#define ROWS 1000
#define COLS 1000

void initialize_matrix_vector(std::vector<std::vector<double>>& matrix, std::vector<double>& vector) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    matrix.resize(ROWS, std::vector<double>(COLS));
    vector.resize(COLS);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    for (int i = 0; i < COLS; ++i) {
        vector[i] = dis(gen);
    }
}

void multiply_sequential(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector, std::vector<double>& result) {
    result.resize(ROWS);
    for (int i = 0; i < ROWS; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < COLS; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void multiply_parallel(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector, std::vector<double>& result) {
    result.resize(ROWS);
    #pragma omp parallel for
    for (int i = 0; i < ROWS; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < COLS; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

int main() {
    std::vector<std::vector<double>> matrix;
    std::vector<double> vector, result_seq, result_par;
    initialize_matrix_vector(matrix, vector);

    auto start_seq = std::chrono::high_resolution_clock::now();
    multiply_sequential(matrix, vector, result_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
    std::cout << "Sequential time: " << seq_time << " seconds\n";

    auto start_par = std::chrono::high_resolution_clock::now();
    multiply_parallel(matrix, vector, result_par);
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end_par - start_par).count();
    std::cout << "Parallel time: " << par_time << " seconds\n";

    bool correct = true;
    for (int i = 0; i < ROWS; ++i) {
        if (std::abs(result_seq[i] - result_par[i]) > 1e-6) {
            correct = false;
            break;
        }
    }
    std::cout << "Results match: " << (correct ? "Yes" : "No") << "\n";

    return 0;
}