#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

#define SIZE 10000

// Инициализация массива случайными значениями
void initialize_array(std::vector<int>& arr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000);

    arr.resize(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        arr[i] = dis(gen);
    }
}

// Последовательная сортировка чётно-нечётной перестановкой
void odd_even_sort_sequential(std::vector<int>& arr) {
    for (int phase = 0; phase < SIZE; ++phase) {
        // Чётная фаза
        if (phase % 2 == 0) {
            for (int i = 1; i < SIZE - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }
        // Нечётная фаза
        else {
            for (int i = 0; i < SIZE - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

// Параллельная сортировка чётно-нечётной перестановкой
void odd_even_sort_parallel(std::vector<int>& arr) {
    for (int phase = 0; phase < SIZE; ++phase) {
        // Чётная фаза
        if (phase % 2 == 0) {
            #pragma omp parallel for
            for (int i = 1; i < SIZE - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }
        // Нечётная фаза
        else {
            #pragma omp parallel for
            for (int i = 0; i < SIZE - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

// Проверка, отсортирован ли массив
bool is_sorted(const std::vector<int>& arr) {
    for (int i = 0; i < SIZE - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<int> arr_seq, arr_par;

    // Инициализация массивов
    initialize_array(arr_seq);
    arr_par = arr_seq; // Копия для параллельной сортировки

    // Последовательная сортировка
    auto start_seq = std::chrono::high_resolution_clock::now();
    odd_even_sort_sequential(arr_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
    std::cout << "Sequential sort time: " << seq_time << " seconds\n";
    std::cout << "Sequential result sorted: " << (is_sorted(arr_seq) ? "Yes" : "No") << "\n";

    // Параллельная сортировка
    auto start_par = std::chrono::high_resolution_clock::now();
    odd_even_sort_parallel(arr_par);
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double>(end_par - start_par).count();
    std::cout << "Parallel sort time: " << par_time << " seconds\n";
    std::cout << "Parallel result sorted: " << (is_sorted(arr_par) ? "Yes" : "No") << "\n";

    return 0;
}