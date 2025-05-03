#ifndef COMMON_H
#define COMMON_H

enum InitType { INIT_ZERO = 0, INIT_ONE = 1, INIT_RANDOM = 2 };

int reduction_cpu(int *input, const int size);
void compare_results(int gpu_result, int cpu_result);
void initialize(int *input, int size, int init_type);

#endif

