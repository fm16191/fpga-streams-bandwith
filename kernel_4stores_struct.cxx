#include "define.hpp"
#include "kernel.hpp"

//
void launcher_4stores_struct(T *d_input, buffers4streams<T> *b4, size_t N, sycl::queue queue)
{
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Start of kernel

            [[intel::fpga_register]] T input, x1, x2, x3, x4;

            for (size_t i = 0; i < N; ++i) {
                input = d_input[i];
                x1 = input + 1;
                x2 = input + 2;
                x3 = input + 3;
                x4 = input + 4;

                b4[i].d1 = x1;
                b4[i].d2 = x2;
                b4[i].d3 = x3;
                b4[i].d4 = x4;
            }

            // End of kernel
        });
    });
}
