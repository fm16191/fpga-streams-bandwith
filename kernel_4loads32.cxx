#include "define.hpp"
#include "kernel.hpp"

//
void launcher_4loads(T *d1, T *d2, T *d3, T *d4, T *d_res, size_t N, sycl::queue queue)
{
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
        // Start of kernel

#pragma unroll 32
            for (size_t i = 0; i < N; ++i) {
                T x1 = d1[i];
                T x2 = d2[i];
                T x3 = d3[i];
                T x4 = d4[i];
                d_res[i] = x1 + x2 + x3 + x4;
            }

            // End of kernel
        });
    });
}
