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
CC       		:= ../calcic
LD_x64      	:= ld
LD_calci32  	:= ../linker/calcil/calcil
CFLAGS   		:= -O --nostdlib 	# Enable optimizations, do not use the previously compiled libraries.
LDFLAGS_x64 	:= -r  				# Relocatable flag to join objects without making a binary
LDFLAGS_calci32 := -c				# Create an object file.

# Output names
LIB_OBJ 	:= calcic_libc.o
ENTRY_OBJ 	:= program_entry.o
SRC_DIR    	:= src
BUILD_DIR 	:= build/$(ARCH)

ifeq ($(ARCH),calci32)
# Temporary explicit list of compilable files for calci32
SRCS := \
	$(SRC_DIR)/arch/calci32/long.c
else
# Find .c files in src/, including src/arch/$(ARCH)/ but excluding other arch folders
SRCS := $(shell find $(SRC_DIR) -name "*.c" \( -path "$(SRC_DIR)/arch/$(ARCH)/*" -o ! -path "$(SRC_DIR)/arch/*" \))
endif

PROGRAM_ENTRY_SRC := $(SRC_DIR)/arch/$(ARCH)/program_entry.s

# Map .c files to .o files in a separate build directory
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Default rule
all: $(BUILD_DIR)/$(LIB_OBJ) $(BUILD_DIR)/$(ENTRY_OBJ)

# Rule to link all individual objects into one single object file
$(BUILD_DIR)/$(LIB_OBJ): $(OBJS)
	@echo "Linking all objects into $@"
	@$(LD_$(ARCH)) $(LDFLAGS_$(ARCH)) $(OBJS) -o $@

$(BUILD_DIR)/$(ENTRY_OBJ) : $(PROGRAM_ENTRY_SRC)
	@echo "Creating program entry object file"
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -m $(ARCH) -c $(PROGRAM_ENTRY_SRC) -o $@

# Rule to compile .c files to .o files
# This pattern handles the subdirectories (stdlib, string, etc.)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -m $(ARCH) -c $< -o $@
ifeq ($(ARCH),x64)
# Convert .globl -> .weak
	@objcopy --weaken $@
endif

# Clean up
clean:
	@rm -rf build

.PHONY: all clean