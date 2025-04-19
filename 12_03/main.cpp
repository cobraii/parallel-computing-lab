#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

const int IMAGE_SIZE = 729;  
const int MAX_DEPTH = 5;     

// Рекурсивная функция генерации ковра Серпинского
void draw_sierpinski(cv::Mat& image, int x, int y, int size, int depth) {
    if (depth == 0) return;

    int newSize = size / 3;

    cv::rectangle(image, cv::Rect(x + newSize, y + newSize, newSize, newSize), cv::Scalar(0, 0, 0), cv::FILLED);

    #pragma omp parallel for collapse(2) if (depth > 1)
    for (int dx = 0; dx < 3; ++dx) {
        for (int dy = 0; dy < 3; ++dy) {
            if (dx == 1 && dy == 1) continue; 
            draw_sierpinski(image, x + dx * newSize, y + dy * newSize, newSize, depth - 1);
        }
    }
}

int main() {
    cv::Mat image(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

    double start_time = omp_get_wtime();

    draw_sierpinski(image, 0, 0, IMAGE_SIZE, MAX_DEPTH);

    double end_time = omp_get_wtime();
    std::cout << "Fractal generated in " << (end_time - start_time) << " seconds\n";

    cv::imwrite("sierpinski_carpet.png", image);
    cv::imshow("Sierpinski Carpet", image);
    cv::waitKey(0);
    return 0;
}
