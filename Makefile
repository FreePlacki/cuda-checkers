# Compiler
NVCC        := nvcc
CXXFLAGS    := 
LDFLAGS     :=

# Directories
SRC_DIR     := src
BUILD_DIR   := build
BIN_DIR     := bin

# Output binary name
TARGET      := $(BIN_DIR)/main

# All .cu source files
CU_SRCS     := $(wildcard $(SRC_DIR)/*.cu)

# Object files for all .cu files
CU_OBJS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

all: $(TARGET)

# Compile each .cu into build/*.o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

# Link everything into final binary
$(TARGET): $(CU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

