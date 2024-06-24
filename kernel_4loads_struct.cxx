#include "define.hpp"
#include "kernel.hpp"

//
void launcher_4loads_struct(buffers4streams<T>* b4, T *d_res, size_t N, sycl::queue queue)
{
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Start of kernel
            [[intel::fpga_register]] T x1, x2, x3, x4;

            for (size_t i = 0; i < N; ++i) {
                x1 = b4[i].d1;
                x2 = b4[i].d2;
                x3 = b4[i].d3;
                x4 = b4[i].d4;
                d_res[i] = x1 + x2 + x3 + x4;
            }

            // End of kernel
        });
    });
}
