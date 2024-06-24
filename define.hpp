#ifndef DEFINE_H_
#define DEFINE_H_

#include <sycl/sycl.hpp>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
    #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

// using T = float;
using T = double;

#include <stddef.h>

typedef struct kernel_timer_t {
    double cpu_to_fpga1, cpu_to_fpga2, fpga_compute, fpga_to_cpu;
} kernel_timer_s;

template <typename T> struct buffers4streams {
    T d1, d2, d3, d4;
};

#endif // DEFINE_H_
