#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> arr;
    double seq_sum = 0.0, par_sum = 0.0;
    double seq_time = 0.0, par_time = 0.0;

    if (rank == 0) {
        initialize_array(arr);
        auto start_seq = std::chrono::high_resolution_clock::now();
        seq_sum = sum_sequential(arr);
        auto end_seq = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
        std::cout << "Sequential sum: " << seq_sum << "\n";
        std::cout << "Sequential time: " << seq_time << " seconds\n";
    }

    if (rank == 0) {
        MPI_Bcast(arr.data(), SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        arr.resize(SIZE);
        MPI_Bcast(arr.data(), SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int chunk_size = SIZE / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? SIZE : start + chunk_size;

    auto start_par = std::chrono::high_resolution_clock::now();
    double local_sum = 0.0;
    for (int i = start; i < end; ++i) {
        local_sum += arr[i];
    }
    MPI_Reduce(&local_sum, &par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    auto end_par = std::chrono::high_resolution_clock::now();
    par_time = std::chrono::duration<double>(end_par - start_par).count();

    if (rank == 0) {
        std::cout << "Parallel sum: " << par_sum << "\n";
        std::cout << "Parallel time (" << size << " processes): " << par_time << " seconds\n";
        std::cout << "Results match: " << (std::abs(seq_sum - par_sum) < 1e-6 ? "Yes" : "No") << "\n";
    }

    MPI_Finalize();
    return 0;
}