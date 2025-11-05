CC = gcc
CFLAGS = -O2 -Wall -Iinclude
LDFLAGS =
TARGET = checkers
SRC = src/main.c
OBJDIR = build
BINDIR = bin
OBJ = $(OBJDIR)/$(notdir $(SRC:.c=.o))
EXEC = $(BINDIR)/$(TARGET)

# NVCC = nvcc
# CUDA_SRC = src/evaluate_position.cu
# CUDA_OBJ = $(OBJDIR)/$(notdir $(CUDA_SRC:.cu=.o))
# OBJ += $(CUDA_OBJ)
# LDFLAGS += -lcuda -lcudart

all: $(EXEC)

$(EXEC): $(OBJ)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Uncomment to support CUDA
# $(OBJDIR)/%.o: src/%.cu
# 	@mkdir -p $(OBJDIR)
# 	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

run: $(EXEC)
	./$(EXEC)

