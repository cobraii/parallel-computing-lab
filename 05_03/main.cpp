#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 100 
#define ITERATIONS 10 
#define ALIVE 'X'
#define DEAD '.'

void initialize_grid(char **grid) {
    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            grid[i][j] = (rand() % 2) ? ALIVE : DEAD;
        }
    }
}

int count_neighbors(char **grid, int x, int y) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int ni = (x + i + SIZE) % SIZE;
            int nj = (y + j + SIZE) % SIZE;
            if (grid[ni][nj] == ALIVE) count++;
        }
    }
    return count;
}

void update_grid(char **current, char **next) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int neighbors = count_neighbors(current, i, j);
            if (current[i][j] == ALIVE) {
                next[i][j] = (neighbors == 2 || neighbors == 3) ? ALIVE : DEAD;
            } else {
                next[i][j] = (neighbors == 3) ? ALIVE : DEAD;
            }
        }
    }
}

void print_grid(char **grid) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%c", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void swap_grids(char ***current, char ***next) {
    char **temp = *current;
    *current = *next;
    *next = temp;
}

int main() {
    char **grid = (char **)malloc(SIZE * sizeof(char *));
    char **next_grid = (char **)malloc(SIZE * sizeof(char *));
    if (!grid || !next_grid) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < SIZE; i++) {
        grid[i] = (char *)malloc(SIZE * sizeof(char));
        next_grid[i] = (char *)malloc(SIZE * sizeof(char));
        if (!grid[i] || !next_grid[i]) {
            fprintf(stderr, "Memory allocation failed for row %d\n", i);
            return 1;
        }
    }

    initialize_grid(grid);

    double start_time = omp_get_wtime();

    // Основной цикл
    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("Starting iteration %d\n", iter);
        if (iter % 5 == 0) { 
            system("cls"); 
            printf("Iteration %d:\n", iter);
            print_grid(grid);
        }
        update_grid(grid, next_grid);
        swap_grids(&grid, &next_grid);
    }

    double end_time = omp_get_wtime();
    printf("Execution time: %f seconds\n", end_time - start_time);

    for (int i = 0; i < SIZE; i++) {
        free(grid[i]);
        free(next_grid[i]);
    }
    free(grid);
    free(next_grid);

    return 0;
}