# **************************************************************************************************
# Use it to build the calcic Standard Libraries.
# Written by dabecart, 2026.
# 
# Call it with "make" to build (defaults to ARCH=x64), "make ARCH=calci32" for calci32,
# or use "make clean" to clean the temporary built files.
# **************************************************************************************************

# Default architecture if none is specified
ARCH ?= x64

# Variables
CC       := ../calcic
LD       := ld
CFLAGS   := -O --nostdlib 	# Enable optimizations, do not use the previously compiled libraries.
LDFLAGS  := -r  			# Relocatable flag to join objects without making a binary

# Output names
TARGET_OBJ := calcic_libc.o
SRC_DIR    := src
OBJ_DIR    := build/$(ARCH)

ifeq ($(ARCH),calci32)
# Temporary explicit list of compilable files for calci32
SRCS := \
    src/arch/calci32/long.c
else
# Find .c files in src/, including src/arch/$(ARCH)/ but excluding other arch folders
SRCS := $(shell find $(SRC_DIR) -name "*.c" \( -path "$(SRC_DIR)/arch/$(ARCH)/*" -o ! -path "$(SRC_DIR)/arch/*" \))
endif

# Map .c files to .o files in a separate build directory
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Default rule
all: $(TARGET_OBJ)

# Rule to link all individual objects into one single object file
$(TARGET_OBJ): $(OBJS)
	@echo "Linking all objects into $@ for architecture: $(ARCH)"
	@$(LD) $(LDFLAGS) $(OBJS) -o $@

# Rule to compile .c files to .o files
# This pattern handles the subdirectories (stdlib, string, etc.)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -m $(ARCH) -c $< -o $@
# Convert .globl -> .weak
	@objcopy --weaken $@

# Clean up
clean:
	rm -rf build $(TARGET_OBJ)

.PHONY: all clean