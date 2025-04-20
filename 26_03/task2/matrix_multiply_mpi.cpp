#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

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

void print_matrix_part(const Matrix& mat, int rows, int cols, int limit = 5) {
    for (int i = 0; i < std::min(rows, limit); ++i) {
        for (int j = 0; j < std::min(cols, limit); ++j) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix A, B, C_seq, C_par;
    double seq_time = 0.0, par_time = 0.0;

    if (rank == 0) {
        initialize_matrix(A, M, N);
        initialize_matrix(B, N, P);
        initialize_matrix(C_seq, M, P);
        initialize_matrix(C_par, M, P);

        auto start_seq = std::chrono::high_resolution_clock::now();
        multiply_sequential(A, B, C_seq);
        auto end_seq = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
        std::cout << "Sequential time: " << seq_time << " seconds\n";
        std::cout << "Sequential result (first 5x5):\n";
        print_matrix_part(C_seq, M, P);
    }

    std::vector<double> B_flat(N * P);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < P; ++j) {
                B_flat[i * P + j] = B[i][j];
            }
        }
    }
    MPI_Bcast(B_flat.data(), N * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        B.resize(N, std::vector<double>(P));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < P; ++j) {
                B[i][j] = B_flat[i * P + j];
            }
        }
    }

    int rows_per_process = M / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? M : start_row + rows_per_process;
    std::vector<std::vector<double>> local_A(end_row - start_row, std::vector<double>(N));

    if (rank == 0) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < N; ++j) {
                local_A[i - start_row][j] = A[i][j];
            }
        }
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_process;
            int p_end = (p == size - 1) ? M : p_start + rows_per_process;
            for (int i = p_start; i < p_end; ++i) {
                MPI_Send(A[i].data(), N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < end_row - start_row; ++i) {
            MPI_Recv(local_A[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    std::vector<std::vector<double>> local_C(end_row - start_row, std::vector<double>(P));
    auto start_par = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < end_row - start_row; ++i) {
        for (int j = 0; j < P; ++j) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    par_time = std::chrono::duration<double>(end_par - start_par).count();

    if (rank == 0) {
        C_par.resize(M, std::vector<double>(P));
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < P; ++j) {
                C_par[i][j] = local_C[i - start_row][j];
            }
        }
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_process;
            int p_end = (p == size - 1) ? M : p_start + rows_per_process;
            for (int i = p_start; i < p_end; ++i) {
                MPI_Recv(C_par[i].data(), P, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = 0; i < end_row - start_row; ++i) {
            MPI_Send(local_C[i].data(), P, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        std::cout << "Parallel time (" << size << " processes): " << par_time << " seconds\n";
        std::cout << "Parallel result (first 5x5):\n";
        print_matrix_part(C_par, M, P);
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
    }

    MPI_Finalize();
    return 0;
}