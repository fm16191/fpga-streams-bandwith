#ifndef KERNEL_H_
#define KERNEL_H_

#include "define.hpp"

void launcher_loads(T *d1, T *d2, T *d3, T *d4, T *d5, T *d6, T *d7, T *d8, T *d_res, size_t N, sycl::queue queue);
void launcher_stores(T *d_input, T *d1, T *d2, T *d3, T *d4, T *d5, T *d6, T *d7, T *d8, size_t N, sycl::queue queue);

double launcher_loads_profiling(T *d1, T *d2, T *d3, T *d4, T *d5, T *d6, T *d7, T *d8, T *d_res, size_t N, sycl::queue queue);
double launcher_stores_profiling(T *d_input, T *d1, T *d2, T *d3, T *d4, T *d5, T *d6, T *d7, T *d8, size_t N, sycl::queue queue);

void launcher_4loads(T *d1, T *d2, T *d3, T *d4, T *d_res, size_t N, sycl::queue queue);
void launcher_4stores(T *d_input, T *d1, T *d2, T *d3, T *d4, size_t N, sycl::queue queue);

void launcher_4loads_struct(buffers4streams <T> *b4, T *d_res, size_t N, sycl::queue queue);
void launcher_4stores_struct(T *d_input, buffers4streams <T> *b4, size_t N, sycl::queue queue);

void launcher_5loads(T *d1, T *d2, T *d3, T *d4, T *d5, T *d_res, size_t N, sycl::queue queue);
void launcher_5stores(T *d_input, T *d1, T *d2, T *d3, T *d4, T *d5, size_t N, sycl::queue queue);

#endif // KERNEL_H_
