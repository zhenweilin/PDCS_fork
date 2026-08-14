# PDCS GPU build targets
# ======================
#
# Production solver build (no diagnostic counter writes):
#
#   make rebuild-gpu CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
#
# Root-finding/projection-work diagnostic build (separate *_profile.ptx files):
#
#   make rebuild-profile CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
#
# `rebuild-profile` defines PDCS_PROFILE_ROOT_SEARCH and compiles the wrapper
# sources `*_block_proj_profile.cu`. Those wrappers enable the counter macros
# used by exp_proj.cu and the SOC/RSOC projection code. The resulting files are
# separate from the production PTX files:
#
#   moderate_block_proj_profile.ptx
#   massive_block_proj_profile.ptx
#   sufficient_block_proj_profile.ptx
#
# Building the diagnostic targets does not replace the normal solver PTX.
# Diagnostic PTX performs extra global-memory/atomic counter writes and MUST
# NOT be used for publication runtime measurements. Use `rebuild-gpu` and the
# normal projection API for timings; use `rebuild-profile` only with
# benchmark/R3.5_projection_work_profile.jl for projection-work statistics.
#
# Override CUDA_HOME to select a toolkit independently of the system-wide
# /usr/local/cuda symlink. ARCH=sm_90 is H100; ARCH=sm_80 is A100.
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_90
LINEINFO ?=
CFLAGS = -Xcompiler -fPIC
PTXFLAGS = -ptx
SHAREDFLAGS = -shared
RPATHFLAGS = -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
OUTPUT_DIR = .

# Targets
PTX_TARGET1 = $(OUTPUT_DIR)/moderate_block_proj.ptx
PTX_TARGET2 = $(OUTPUT_DIR)/massive_block_proj.ptx
PTX_TARGET3 = $(OUTPUT_DIR)/sufficient_block_proj.ptx
PTX_TARGET4 = $(OUTPUT_DIR)/utils.ptx
SO_TARGET = $(OUTPUT_DIR)/libfew_block_proj.so
PROFILE_TARGET1 = $(OUTPUT_DIR)/moderate_block_proj_profile.ptx
PROFILE_TARGET2 = $(OUTPUT_DIR)/massive_block_proj_profile.ptx
PROFILE_TARGET3 = $(OUTPUT_DIR)/sufficient_block_proj_profile.ptx

# Source files
SRC1 = moderate_block_proj.cu
SRC2 = massive_block_proj.cu
SRC3 = few_block_proj.cu
SRC4 = sufficient_block_proj.cu
SRC5 = utils.cu
# The default target is the production build. Profile PTX is deliberately not
# part of `all`, so an ordinary `make` cannot accidentally enable diagnostics.
all: $(PTX_TARGET1) $(PTX_TARGET2) $(PTX_TARGET3) $(PTX_TARGET4) $(SO_TARGET) 

# Ensure output directory exists
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Rule for moderate_block_proj.ptx
$(PTX_TARGET1): $(SRC1) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC1) -o $(PTX_TARGET1)

# Rule for massive_block_proj.ptx
$(PTX_TARGET2): $(SRC2) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC2) -o $(PTX_TARGET2)

# Rule for sufficient_block_proj.ptx
$(PTX_TARGET3): $(SRC4) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC4) -o $(PTX_TARGET3)

# Rule for utils.ptx
$(PTX_TARGET4): $(SRC5) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC5) -o $(PTX_TARGET4)

# Rule for libfew_block_proj.so
$(SO_TARGET): $(SRC3) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

# Always rebuild production artifacts. This is the preferred target when
# switching CUDA toolkits because timestamps alone cannot encode CUDA_HOME.
.PHONY: help rebuild-gpu rebuild-few rebuild-profile print-config

help:
	@echo "PDCS CUDA build targets"
	@echo "  make rebuild-gpu     Production PTX/shared library; use for solver timings"
	@echo "  make rebuild-few     Production gridWise shared library only"
	@echo "  make rebuild-profile Separate diagnostic PTX with root-search counters"
	@echo "  make print-config    Show the selected CUDA toolkit and GPU architecture"
	@echo ""
	@echo "Examples:"
	@echo "  make rebuild-gpu CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90"
	@echo "  make rebuild-profile CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90"
	@echo ""
	@echo "Do not use *_profile.ptx for publication runtime measurements."

rebuild-gpu: | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC1) -o $(PTX_TARGET1)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC2) -o $(PTX_TARGET2)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC4) -o $(PTX_TARGET3)
	$(NVCC) -arch=$(ARCH) -O3 $(LINEINFO) $(PTXFLAGS) $(SRC5) -o $(PTX_TARGET4)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

rebuild-few: | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

# Compile diagnostic wrappers into separate output files. This target neither
# overwrites nor relinks the production PTX/shared-library artifacts.
rebuild-profile: | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) -DPDCS_PROFILE_ROOT_SEARCH=1 -DPDCS_PROFILE_MAPPING_BLOCK=1 moderate_block_proj_profile.cu -o $(PROFILE_TARGET1)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) -DPDCS_PROFILE_ROOT_SEARCH=1 -DPDCS_PROFILE_MAPPING_THREAD=1 massive_block_proj_profile.cu -o $(PROFILE_TARGET2)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) -DPDCS_PROFILE_ROOT_SEARCH=1 -DPDCS_PROFILE_MAPPING_WARP=1 sufficient_block_proj_profile.cu -o $(PROFILE_TARGET3)

print-config:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "NVCC=$(NVCC)"
	@echo "ARCH=$(ARCH)"
	@$(NVCC) --version | tail -n 1

# Clean up generated files
clean:
	rm -rf $(OUTPUT_DIR)/*.ptx $(OUTPUT_DIR)/*.so
