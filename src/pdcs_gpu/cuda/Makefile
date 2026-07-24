# Build configuration. Override these on the make command line to select a
# toolkit independently of the system-wide /usr/local/cuda symlink, e.g.
#   make rebuild-few CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_90
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

# Source files
SRC1 = moderate_block_proj.cu
SRC2 = massive_block_proj.cu
SRC3 = few_block_proj.cu
SRC4 = sufficient_block_proj.cu
SRC5 = utils.cu
# Default target
all: $(PTX_TARGET1) $(PTX_TARGET2) $(PTX_TARGET3) $(PTX_TARGET4) $(SO_TARGET) 

# Ensure output directory exists
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Rule for moderate_block_proj.ptx
$(PTX_TARGET1): $(SRC1) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC1) -o $(PTX_TARGET1)

# Rule for massive_block_proj.ptx
$(PTX_TARGET2): $(SRC2) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC2) -o $(PTX_TARGET2)

# Rule for sufficient_block_proj.ptx
$(PTX_TARGET3): $(SRC4) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC4) -o $(PTX_TARGET3)

# Rule for utils.ptx
$(PTX_TARGET4): $(SRC5) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC5) -o $(PTX_TARGET4)

# Rule for libfew_block_proj.so
$(SO_TARGET): $(SRC3) | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

# Always rebuild only the shared library. This is the preferred target when
# switching CUDA toolkits because timestamps alone cannot encode CUDA_HOME.
.PHONY: rebuild-gpu rebuild-few print-config
rebuild-gpu: | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC1) -o $(PTX_TARGET1)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC2) -o $(PTX_TARGET2)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC4) -o $(PTX_TARGET3)
	$(NVCC) -arch=$(ARCH) -O3 $(PTXFLAGS) $(SRC5) -o $(PTX_TARGET4)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

rebuild-few: | $(OUTPUT_DIR)
	$(NVCC) -arch=$(ARCH) -O3 $(CFLAGS) $(SHAREDFLAGS) $(RPATHFLAGS) $(SRC3) -o $(SO_TARGET) -lcublas

print-config:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "NVCC=$(NVCC)"
	@echo "ARCH=$(ARCH)"
	@$(NVCC) --version | tail -n 1

# Clean up generated files
clean:
	rm -rf $(OUTPUT_DIR)/*.ptx $(OUTPUT_DIR)/*.so
