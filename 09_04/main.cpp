#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITER 1000
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5

int mandelbrot(double real, double imag) {
    std::complex<double> c(real, imag);
    std::complex<double> z(0, 0);
    int iter = 0;
    while (std::norm(z) <= 4.0 && iter < MAX_ITER) {
        z = z * z + c;
        iter++;
    }
    return iter;
}

void compute_mandelbrot_sequential(std::vector<int>& buffer) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = X_MIN + (X_MAX - X_MIN) * x / WIDTH;
            double imag = Y_MIN + (Y_MAX - Y_MIN) * y / HEIGHT;
            buffer[y * WIDTH + x] = mandelbrot(real, imag);
        }
    }
}

void buffer_to_image(const std::vector<int>& buffer, cv::Mat& image) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int iter = buffer[y * WIDTH + x];
            int value = (iter == MAX_ITER) ? 0 : (255 * iter / MAX_ITER);
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, 255);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<int> buffer(HEIGHT * WIDTH, 0);

    double seq_time = 0.0;
    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        compute_mandelbrot_sequential(buffer);
        auto end = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(end - start).count();
        std::cout << "Sequential time: " << seq_time << " seconds\n";
    }

    auto start = std::chrono::high_resolution_clock::now();

    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_process;

    std::vector<int> local_buffer((end_row - start_row) * WIDTH, 0);

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = X_MIN + (X_MAX - X_MIN) * x / WIDTH;
            double imag = Y_MIN + (Y_MAX - Y_MIN) * y / HEIGHT;
            local_buffer[(y - start_row) * WIDTH + x] = mandelbrot(real, imag);
        }
    }

    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        int rows = (i == size - 1) ? (HEIGHT - i * rows_per_process) : rows_per_process;
        recv_counts[i] = rows * WIDTH;
        displs[i] = i * rows_per_process * WIDTH;
    }

    MPI_Gatherv(local_buffer.data(), local_buffer.size(), MPI_INT,
                buffer.data(), recv_counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        std::cout << "Parallel time (" << size << " processes): " << par_time << " seconds\n";
        buffer_to_image(buffer, image);
        cv::imwrite("mandelbrot.png", image);
        cv::imshow("Mandelbrot Set", image);
        cv::waitKey(0);
    }

    MPI_Finalize();
    return 0;
}