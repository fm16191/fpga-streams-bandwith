#include "define.hpp"
#include "kernel.hpp"

//
void launcher_4stores(T *d_input, T *d1, T *d2, T *d3, T *d4, size_t N, sycl::queue queue)
{
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
        // Start of kernel

            for (size_t i = 0; i < N; ++i) {
                T input = d_input[i];
                T x1 = input + 1;
                T x2 = input + 2;
                T x3 = input + 3;
                T x4 = input + 4;

                d1[i] = x1;
                d2[i] = x2;
                d3[i] = x3;
                d4[i] = x4;
            }

            // End of kernel
        });
    });
}
