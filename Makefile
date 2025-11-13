CC = gcc
CFLAGS = -g -Wall -Wextra -DFORMATTING -Iinclude
LDFLAGS = 
TARGET = checkers
SRCDIR = src
OBJDIR = build
BINDIR = bin

SRC = $(wildcard $(SRCDIR)/*.c)
OBJ = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRC))
EXEC = $(BINDIR)/$(TARGET)

all: $(EXEC)

$(EXEC): $(OBJ)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

run: $(EXEC)
	./$(EXEC)

