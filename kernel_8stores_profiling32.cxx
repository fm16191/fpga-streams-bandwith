#include "define.hpp"
#include "kernel.hpp"

//
static inline __attribute__((always_inline, pure)) double get_t_ns(sycl::event e)
{
    return static_cast<double>(e.template get_profiling_info<sycl::info::event_profiling::command_end>()) -
           static_cast<double>(e.template get_profiling_info<sycl::info::event_profiling::command_start>());
}

//
double launcher_stores_profiling(T *d_input, T *d1, T *d2, T *d3, T *d4, T *d5, T *d6, T *d7, T *d8, size_t N, sycl::queue queue)
{
    sycl::event fpga_compute_event = queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
        // Start of kernel

#pragma unroll 32
            for (size_t i = 0; i < N; ++i) {
                T input = d_input[i];
                T x1 = input + 1;
                T x2 = input + 2;
                T x3 = input + 3;
                T x4 = input + 4;
                T x5 = input + 5;
                T x6 = input + 6;
                T x7 = input + 7;
                T x8 = input + 8;

                d1[i] = x1;
                d2[i] = x2;
                d3[i] = x3;
                d4[i] = x4;
                d5[i] = x5;
                d6[i] = x6;
                d7[i] = x7;
                d8[i] = x8;
            }

            // End of kernel
        });
    });

    double fpga_compute = get_t_ns(fpga_compute_event);
    return fpga_compute;
}
