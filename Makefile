CXX := icpx -fsycl
CXXFLAGS := -O2 -g -std=c++2b -fPIC -march=native -Wall -Wextra -Weverything -Wno-c++98-compat -Wno-undef -Wno-unused-function -Wno-unsafe-buffer-usage
LDFLAGS := -Wl,--unresolved-symbols=ignore-in-object-files # -Dlauncher_p70=atexit
# -fsafe-buffer-usage-suggestions # only in oneAPI 2024.0
FLAGS_FPGA := -fintelfpga

SRC := main.cxx
OBJ := $(SRC:.cxx=.o)
KERNEL_SRC := 
KERNEL_OBJ := $(KERNEL_SRC:.cxx=.o)

BUILD_TYPE :=
OPTION :=

BOARD_NAME := ia840f:ofs_ia840fr0

TARGET_NAME := $(KERNEL_SRC:.cxx=)
CPU_EXE_NAME := $(TARGET_NAME).cpu
EMU_EXE_NAME := $(TARGET_NAME).fpga_emu
SIMU_EXE_NAME := $(TARGET_NAME).fpga_simu
REPORT_NAME := $(TARGET_NAME)_report.a
FPGA_EXE_NAME := $(TARGET_NAME).fpga

.PHONY: cpu fpga_emu run_fpga_emu fpga_simu run_fpga_simu report fpga run_fpga

%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(BUILD_TYPE) -c $< -o $@ $(OPTION)

# CPU
cpu: $(OBJ) $(KERNEL_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(CPU_EXE_NAME) $(LDFLAGS)
run_cpu:
	./$(CPU_EXE_NAME)


# Emulator
fpga_emu: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_EMULATOR=1
fpga_emu: $(OBJ) $(KERNEL_OBJ)
	$(CXX) $(BUILD_TYPE) $^ -o $(EMU_EXE_NAME) $(LDFLAGS)
run_fpga_emu:
	./$(EMU_EXE_NAME)


# Simulator
fpga_simu: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_SIMULATOR=1 -Xssimulation
fpga_simu: $(OBJ)
	$(CXX) $(BUILD_TYPE) $^ -o $(SIMU_EXE_NAME) $(LDFLAGS)
run_fpga_simu:
	./$(SIMU_EXE_NAME)

# Report
report: BUILD_TYPE := $(FLAGS_FPGA) -fsycl-link=early -Xshardware -DFPGA_HARDWARE=1
report: $(KERNEL_SRC)
	$(CXX) $(CXXFLAGS) $(BUILD_TYPE) $^ -o $(REPORT_NAME)


# Hardware
KERNEL_FPGA_SO := $(KERNEL_SRC:.cxx=.so)
$(KERNEL_FPGA_SO): BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1
$(KERNEL_FPGA_SO): $(KERNEL_OBJ)
	$(CXX) $(CXXFLAGS) $(BUILD_TYPE) -shared -Xsprofile -Xshardware -Xsparallel=3 -Xstarget=$(BOARD_NAME) -fsycl-link=image $^ -o $@ $(OPTION)

KERNEL_EMIT_BC := $(KERNEL_SRC:.cxx=.bc)
bc: $(KERNEL_EMIT_BC)
$(KERNEL_EMIT_BC): BUILD_TYPE := -emit-llvm -flto
$(KERNEL_EMIT_BC): $(KERNEL_OBJ)
	icpx -fsycl -emit-llvm -flto -c $^ -o $@1
	llvm-dis -o $@2 $^
	icpx -fsycl -emit-llvm -flto -c $(KERNEL_SRC) -o $@3
	llvm-dis -o $@4 $(KERNEL_SRC)
	$(CXX) $(CXXFLAGS) -emit-llvm -flto -c $(KERNEL_SRC) -o $@5
	$(CXX) $(CXXFLAGS) $(FLAGS_FPGA) -DFPGA_HARDWARE=1 -emit-llvm -flto -c $(KERNEL_SRC) -o $@6
	icpx -fno-sycl $(CXXFLAGS) -emit-llvm -flto -c $(KERNEL_SRC) -o $@7
	icpx -fno-sycl $(CXXFLAGS) $(FLAGS_FPGA) -DFPGA_HARDWARE=1 -emit-llvm -flto -c $(KERNEL_SRC) -o $@8


fpga: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1
recompile_fpga: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1

fpga: $(OBJ) $(KERNEL_FPGA_SO)
	$(CXX) $(BUILD_TYPE) $^ -o $(FPGA_EXE_NAME) $(OPTION) $(LDFLAGS)
recompile_fpga: $(OBJ)
	$(CXX) $(BUILD_TYPE) $^ $(KERNEL_FPGA_SO) -o $(FPGA_EXE_NAME) $(OPTION) $(LDFLAGS)
run_fpga:
	./$(FPGA_EXE_NAME)

clean:
	rm -rf *.o *.d *.out *.mon *.aocr *.aoco *.prj *.cpu *.fpga_emu *.fpga_simu *.a $(FPGA_EXE_NAME)
