#include "define.hpp"
#include "kernel.hpp"

#include <chrono>
#include <cmath>
#include <cmath> // for std::abs
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stddef.h>
#include <string>
#include <sycl/sycl.hpp>
#include <sys/time.h>

#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
    #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

using std::cerr;
using std::cout;
using std::min;
using std::setprecision;
using std::sqrt;
using std::stod;
using std::stoi;

// Defaults
static size_t NB_ITER = 100;

constexpr T tolerance = static_cast<T>(1e-6);

/*** Print device information
 * @param q the oneAPI queue
 */
static void PrintTargetInfo(queue &q)
{
    auto device = q.get_device();
    auto max_block_size = device.get_info<info::device::max_work_group_size>();
    auto max_EU_count = device.get_info<info::device::max_compute_units>();

    cout << " Running on " << device.get_info<info::device::name>() << "\n";
    cout << " The Device Max Work Group Size is : " << max_block_size << "\n";
    cout << " The Device Max EUCount is : " << max_EU_count << "\n";
}

typedef struct results_t {
    double sum, min, max;
    double mean, standard_deviation;
} results_s;

/*** Returns average execution time along with sd and min/max values, given a
 * timers set.
 * @param timers timers pointer
 * @param max_iter maximum number of iterations on timers
 */
static results_s timers_stats(double *timers, size_t max_iter)
{
    double sum = 0.0;
    double min = timers[0];
    double max = timers[0];

    for (size_t i = 0; i < max_iter; i++) {
        sum += timers[i];
        if (timers[i] < min) min = timers[i];
        if (timers[i] > max) max = timers[i];
    }

    double mean = sum / static_cast<double>(max_iter);
    double sum2 = 0.0;

    for (size_t i = 0; i < max_iter; i++) {
        double it = timers[i] - mean;
        sum2 += it * it;
    }
    double standard_deviation = sqrt(sum2 / static_cast<double>(max_iter - 1));

    results_s res = { sum, min, max, mean, standard_deviation };
    return res;
}

/*** Returns average execution time along with sd and min/max values, given a
 * timers set.
 * @param timers timers pointer
 * @param items maximum number of iterations on timers
 * @param name timers's name
 */
static void timers_print(double *timers, size_t items, std::string name)
{
    results_s res = timers_stats(timers, items);
    cout << "-----------------------------------------------------------\n" << name << "\n";
    printf("Average execution time : (mean ± σ):     %.1f us ± %.1f us\n", res.mean, res.standard_deviation);
    printf("                         (min … max):    %.1f us … %.1f us\n", res.min, res.max);
    printf("\n");
}

int main(int argc, char *argv[])
{
    char *MODE = nullptr;
    std::string exec(argv[0]);
    (void)argc; // remove warning

    // Get MODE from the executable name
    size_t start_pos = exec.find("kernel_");
    if (start_pos != std::string::npos) {
        start_pos += 7; // Length of "kernel_"
        size_t end_pos = exec.find('.', start_pos);
        if (end_pos != std::string::npos) MODE = strdup(exec.substr(start_pos, end_pos - start_pos).c_str());
    }
    else MODE = strdup("");

    size_t N = 1e7;
    if (argc > 1) N = static_cast<size_t>(atoi(argv[1]));

#if FPGA_EMULATOR
    // Intel extension: FPGA emulator selector on systems without FPGA card.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
    // Intel extension: FPGA simulator selector on systems without FPGA card.
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    // Intel extension: FPGA selector on systems with FPGA card.
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    // The default device selector will select the most performant device.
    auto selector = default_selector_v;
#endif

    cerr << "Creating device queue - loading FPGA design\n";
    // Create the device queue
    auto t1 = std::chrono::high_resolution_clock::now();
    sycl::queue queue(selector, sycl::property::queue::enable_profiling{});
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t_queue = t2 - t1;
    cerr << "FPGA design loaded in " << std::setprecision(2) << t_queue.count() / 1e3 << "s \n";

    auto device = queue.get_device();
    if (!queue.get_device().has(sycl::aspect::queue_profiling)) {
        cerr << "Device does not support profiling." << std::endl;
        // return 1;
    }
    PrintTargetInfo(queue);

    // Allocations
    size_t alloc_size = sizeof(T) * N;
    T *h_input = reinterpret_cast<T *>(malloc(alloc_size));
    T *d_input = sycl::malloc_device<T>(alloc_size, queue);

    T *h_res = reinterpret_cast<T *>(malloc(alloc_size));
    T *d_res = sycl::malloc_device<T>(alloc_size, queue);

    T *h1 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d1 = sycl::malloc_device<T>(alloc_size, queue);

    T *h2 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d2 = sycl::malloc_device<T>(alloc_size, queue);

    T *h3 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d3 = sycl::malloc_device<T>(alloc_size, queue);

    T *h4 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d4 = sycl::malloc_device<T>(alloc_size, queue);

    T *h5 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d5 = sycl::malloc_device<T>(alloc_size, queue);

    T *h6 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d6 = sycl::malloc_device<T>(alloc_size, queue);

    T *h7 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d7 = sycl::malloc_device<T>(alloc_size, queue);

    T *h8 = reinterpret_cast<T *>(malloc(alloc_size));
    T *d8 = sycl::malloc_device<T>(alloc_size, queue);

    // if ((!strncmp(MODE, "4loads_struct", 13)) || (!strncmp(MODE, "4stores_struct", 14)))
    buffers4streams<T> *b4 = new buffers4streams<T>[N];

    // Expected
    T *h_expected_res = reinterpret_cast<T *>(malloc(alloc_size));

    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<T>(i);
        h1[i] = static_cast<T>(i) + static_cast<T>(1);
        h2[i] = static_cast<T>(i) + static_cast<T>(2);
        h3[i] = static_cast<T>(i) + static_cast<T>(3);
        h4[i] = static_cast<T>(i) + static_cast<T>(4);
        h5[i] = static_cast<T>(i) + static_cast<T>(5);
        h6[i] = static_cast<T>(i) + static_cast<T>(6);
        h7[i] = static_cast<T>(i) + static_cast<T>(7);
        h8[i] = static_cast<T>(i) + static_cast<T>(8);
        h_res[i] = static_cast<T>(0);
    }

    // timers allocations
    double *timers_cpu_to_fpga = new double[NB_ITER - 1];
    double *timers_fpga_compute = new double[NB_ITER - 1];
    double *timers_fpga_to_cpu = new double[NB_ITER - 1];

    // Kernel
    auto t1_simu = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < NB_ITER; ++t) {
        struct timespec cpu_to_fpga_t1, cpu_to_fpga_t2, fpga_compute_t1, fpga_compute_t2, fpga_to_cpu_t1,
            fpga_to_cpu_t2;

        double cpu_to_fpga, fpga_compute = 0.0, fpga_to_cpu, fpga_total_compute;

        /* copy cpu to fpga */
        clock_gettime(CLOCK_MONOTONIC, &cpu_to_fpga_t1);

        if (!strncmp(MODE + 1, "loads", 5)) {
            queue.memcpy(d1, h1, alloc_size);
            queue.memcpy(d2, h2, alloc_size);
            queue.memcpy(d3, h3, alloc_size);
            queue.memcpy(d4, h4, alloc_size);
        }
        if (!strncmp(MODE, "5loads", 6)) queue.memcpy(d5, h5, alloc_size);
        if (!strncmp(MODE, "8loads", 6)) {
            queue.memcpy(d5, h5, alloc_size);
            queue.memcpy(d6, h6, alloc_size);
            queue.memcpy(d7, h7, alloc_size);
            queue.memcpy(d8, h8, alloc_size);
        }
        if (!strncmp(MODE + 1, "stores", 6)) {
            queue.memcpy(d_input, h_input, alloc_size);
        }
        queue.wait();

        clock_gettime(CLOCK_MONOTONIC, &cpu_to_fpga_t2);

        /* Computation */
        clock_gettime(CLOCK_MONOTONIC, &fpga_compute_t1);

        // 8 PROFILING
        if (!strncmp(MODE, "8loads_profiling", 16)) {
            fpga_compute = launcher_loads_profiling(d1, d2, d3, d4, d5, d6, d7, d8, d_res, N, queue);
        }
        else if (!strncmp(MODE, "8stores_profiling", 17)) {
            fpga_compute = launcher_stores_profiling(d_input, d1, d2, d3, d4, d5, d6, d7, d8, N, queue);
        }
        // 8
        else if (!strncmp(MODE, "8loads", 6)) {
            launcher_loads(d1, d2, d3, d4, d5, d6, d7, d8, d_res, N, queue);
        }
        else if (!strncmp(MODE, "8stores", 7)) {
            launcher_stores(d_input, d1, d2, d3, d4, d5, d6, d7, d8, N, queue);
        }
        // 4 & STRUCT
        else if (!strncmp(MODE, "4loads_struct", 13)) {
            for (size_t i = 0; i < N; ++i) {
                b4[i].d1 = d1[i];
                b4[i].d2 = d2[i];
                b4[i].d3 = d3[i];
                b4[i].d4 = d4[i];
            }
            clock_gettime(CLOCK_MONOTONIC, &fpga_compute_t1);
            launcher_4loads_struct(b4, d_res, N, queue);
        }
        else if (!strncmp(MODE, "4stores_struct", 14)) {
            for (size_t i = 0; i < N; ++i) {
                b4[i].d1 = d1[i];
                b4[i].d2 = d2[i];
                b4[i].d3 = d3[i];
                b4[i].d4 = d4[i];
            }
            clock_gettime(CLOCK_MONOTONIC, &fpga_compute_t1);
            launcher_4stores_struct(d_input, b4, N, queue);
        }
        // 4
        else if (!strncmp(MODE, "4loads", 6)) {
            launcher_4loads(d1, d2, d3, d4, d_res, N, queue);
        }
        else if (!strncmp(MODE, "4stores", 7)) {
            launcher_4stores(d_input, d1, d2, d3, d4, N, queue);
        }
        // 5
        else if (!strncmp(MODE, "5loads", 6)) {
            launcher_5loads(d1, d2, d3, d4, d5, d_res, N, queue);
        }
        else if (!strncmp(MODE, "5stores", 7)) {
            launcher_5stores(d_input, d1, d2, d3, d4, d5, N, queue);
        }
        queue.wait();
        clock_gettime(CLOCK_MONOTONIC, &fpga_compute_t2);

        if (!strncmp(MODE, "4stores_struct", 14)) {
            for (size_t i = 0; i < N; ++i) {
                d1[i] = b4[i].d1;
                d2[i] = b4[i].d2;
                d3[i] = b4[i].d3;
                d4[i] = b4[i].d4;
            }
        }

        /* copy fpga to cpu */
        clock_gettime(CLOCK_MONOTONIC, &fpga_to_cpu_t1);
        if (!strncmp(MODE + 1, "loads", 5)) {
            queue.memcpy(h_res, d_res, alloc_size);
        }
        if (!strncmp(MODE + 1, "stores", 6)) {
            queue.memcpy(h1, d1, alloc_size);
            queue.memcpy(h2, d2, alloc_size);
            queue.memcpy(h3, d3, alloc_size);
            queue.memcpy(h4, d4, alloc_size);
        }
        if (!strncmp(MODE, "5stores", 7)) queue.memcpy(h5, d5, alloc_size);
        if (!strncmp(MODE, "8stores", 7)) {
            queue.memcpy(h5, d5, alloc_size);
            queue.memcpy(h6, d6, alloc_size);
            queue.memcpy(h7, d7, alloc_size);
            queue.memcpy(h8, d8, alloc_size);
        }
        queue.wait();
        clock_gettime(CLOCK_MONOTONIC, &fpga_to_cpu_t2);

        cpu_to_fpga = static_cast<double>(cpu_to_fpga_t2.tv_sec - cpu_to_fpga_t1.tv_sec) * 1e6 +
                      static_cast<double>(cpu_to_fpga_t2.tv_nsec - cpu_to_fpga_t1.tv_nsec) / 1e3;
        if (strncmp(MODE, "8loads_profiling", 16) && strncmp(MODE, "8stores_profiling",
                                                             17)) // if we aren't using internal profiling
            fpga_compute = static_cast<double>(fpga_compute_t2.tv_sec - fpga_compute_t1.tv_sec) * 1e6 +
                           static_cast<double>(fpga_compute_t2.tv_nsec - fpga_compute_t1.tv_nsec) / 1e3;
        else fpga_compute /= 1e3;
        fpga_to_cpu = static_cast<double>(fpga_to_cpu_t2.tv_sec - fpga_to_cpu_t1.tv_sec) * 1e6 +
                      static_cast<double>(fpga_to_cpu_t2.tv_nsec - fpga_to_cpu_t1.tv_nsec) / 1e3;

        fpga_total_compute = cpu_to_fpga + fpga_compute + fpga_to_cpu;
        printf(" compute time : %.2f ms (%.2f, %.0f us, %.2f)\n", fpga_total_compute / 1e3, cpu_to_fpga / 1e3,
               fpga_compute, fpga_to_cpu / 1e3);

        timers_cpu_to_fpga[t] = cpu_to_fpga;
        timers_fpga_compute[t] = fpga_compute;
        timers_fpga_to_cpu[t] = fpga_to_cpu;
    }

    size_t indices[]{ 0, 1, 2, N / 2 - 1, N / 2, N / 2 + 1, N - 1 };
    constexpr size_t indices_size = sizeof(indices) / sizeof(size_t);

    printf("\n");
    printf("MODE = %s\n", MODE);
    printf("items : %ld\n", N);

    for (size_t i = 0; i < indices_size; i++) {
        size_t j = indices[i];

        T tmp = 0;
        // 8
        if (!strncmp(MODE, "8stores", 7)) {
            h_expected_res[j] = 8 * h_input[j] + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;
            tmp = h1[j] + h2[j] + h3[j] + h4[j] + h5[j] + h6[j] + h7[j] + h8[j];
        }
        else if (!strncmp(MODE, "8loads", 6)) {
            h_expected_res[j] = 8 * h_input[j] + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;
            tmp = h_res[j];
        }
        // 4
        else if (!strncmp(MODE, "4stores", 7)) {
            h_expected_res[j] = 4 * h_input[j] + 1 + 2 + 3 + 4;
            tmp = h1[j] + h2[j] + h3[j] + h4[j];
        }
        else if (!strncmp(MODE, "4loads", 6)) {
            h_expected_res[j] = 4 * h_input[j] + 1 + 2 + 3 + 4;
            tmp = h_res[j];
        }
        // 5
        else if (!strncmp(MODE, "5loads", 6)) {
            h_expected_res[j] = 5 * h_input[j] + 1 + 2 + 3 + 4 + 5;
            tmp = h_res[j];
        }
        else if (!strncmp(MODE, "5stores", 7)) {
            h_expected_res[j] = 5 * h_input[j] + 1 + 2 + 3 + 4 + 5;
            tmp = h1[j] + h2[j] + h3[j] + h4[j] + h5[j];
        }

        cout << "[" << j << "] res : " << tmp << " == " << h_expected_res[j];
        if (std::abs(tmp - h_expected_res[j]) < tolerance) cout << " OK\n";
        else cout << " NO\n";
    }

    auto t2_simu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t_simu = t2_simu - t1_simu;

    timers_print(timers_cpu_to_fpga, NB_ITER - 1, "-- copy CPU to FPGA --");
    timers_print(timers_fpga_compute, NB_ITER - 1, "-- FPGA compute time --");
    timers_print(timers_fpga_to_cpu, NB_ITER - 1, "-- copy FPGA to CPU --");

    printf("Simulation execution time : %f sec\n", t_simu.count() / 1e3);
    printf("Iteration execution time  : %f msec\n", t_simu.count() / static_cast<double>(NB_ITER - 1));

    delete[] timers_cpu_to_fpga;
    delete[] timers_fpga_compute;
    delete[] timers_fpga_to_cpu;

    return 0;
}
