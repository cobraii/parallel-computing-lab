#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

#define M 1000
#define N 1000
#define P 1000

using Matrix = std::vector<std::vector<double>>;

void initialize_matrix(Matrix& mat, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    mat.resize(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = dis(gen);
        }
    }
}

void multiply_sequential(const Matrix& A, const Matrix& B, Matrix& C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiply_parallel(const Matrix& A, const Matrix& B, Matrix& C) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    Matrix A, B, C_seq, C_par;
    
    initialize_matrix(A, M, N);
    initialize_matrix(B, N, P);
    initialize_matrix(C_seq, M, P);
    initialize_matrix(C_par, M, P);

    auto start_seq = std::chrono::high_resolution_clock::now();
    multiply_sequential(A, B, C_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
    std::cout << "Sequential multiplication time: " << seq_time << " seconds\n";

    auto start_par = std::chrono::high_resolution_clock::now();
    multiply_parallel(A, B, C_par);
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end_par - start_par).count();
    std::cout << "Parallel multiplication time: " << par_time << " seconds\n";

    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            if (std::abs(C_seq[i][j] - C_par[i][j]) > 1e-6) {
                correct = false;
                break;
            }
        }
    }
    std::cout << "Results match: " << (correct ? "Yes" : "No") << "\n";

    return 0;
}