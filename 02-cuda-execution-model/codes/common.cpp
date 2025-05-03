#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int reduction_cpu(int *input, const int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
}

void compare_results(int gpu_result, int cpu_result) {
    printf("GPU RESULT : %d, CPU RESULT : %d\n", gpu_result, cpu_result);
    if (gpu_result == cpu_result) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
}

void initialize(int *input, int size, int init_type) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        switch (init_type) {
            case INIT_ZERO: input[i] = 0; break;
            case INIT_ONE: input[i] = 1; break;
            case INIT_RANDOM: input[i] = rand() % 100; break;
        }
    }
}

