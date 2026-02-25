# **************************************************************************************************
# Use it to build the calcic Standard Libraries.
# Written by dabecart, 2026.
# 
# Call it with "make" to build, or use "make clean" to clean the temporary built files.
# **************************************************************************************************

# Variables
CC       := ../calcic
LD       := ld
CFLAGS   := -O --nostdlib # Enable optimizations, do not use the previously compiled libraries.
LDFLAGS  := -r  # Relocatable flag to join objects without making a binary

# Output names
TARGET_OBJ := lib.o
SRC_DIR    := src
OBJ_DIR    := build

# Find all .c files in all subdirectories of src/
SRCS := $(shell find $(SRC_DIR) -name "*.c")

# Map .c files to .o files in a separate build directory
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Default rule
all: $(TARGET_OBJ)

# Rule to link all individual objects into one single object file
$(TARGET_OBJ): $(OBJS)
	@echo "Linking all objects into $@"
	@$(LD) $(LDFLAGS) $(OBJS) -o $@

# Rule to compile .c files to .o files
# This pattern handles the subdirectories (stdlib, string, etc.)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET_OBJ)

.PHONY: all clean