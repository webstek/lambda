# ****************************************************************************
# Makefile for Lambda path tracer (courtesy Claud)
# ****************************************************************************

CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra
DEBUG_FLAGS := -g -DDEBUG -fopenmp -mavx2
RELEASE_FLAGS := -O3 -fopenmp -march=native
SUPPRESS_FLAGS := -Wno-deprecated-literal-operator
TRAP_FLAGS := -fsanitize=undefined -fno-sanitize-recover=undefined -fsignaling-nans -ftrapping-math

SRC_DIR := src
INCLUDE_DIR := include
LIB_DIR := lib
BIN_DIR := bin
BUILD_DIR := build

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp) \
           $(LIB_DIR)/LodePNG/lodepng.cpp \
           $(LIB_DIR)/fast_obj.c
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter $(SRC_DIR)/%,$(SOURCES))) \
           $(BUILD_DIR)/lodepng.o \
           $(BUILD_DIR)/fast_obj.o

# Include paths - find all subdirectories in include/ and lib/
INCLUDES :=-I$(INCLUDE_DIR) $(addprefix -I,$(shell find $(INCLUDE_DIR) -mindepth 1 -type d))
INCLUDES +=-I$(LIB_DIR) $(addprefix -I,$(shell find $(LIB_DIR) -mindepth 1 -type d))

# Target executable
TARGET := $(BIN_DIR)/lambda

# Default target (debug)
.PHONY: all debug release clean

all: debug

debug: CXXFLAGS += $(DEBUG_FLAGS) $(SUPPRESS_FLAGS) $(TRAP_FLAGS)
debug: $(TARGET)

release: CXXFLAGS += $(RELEASE_FLAGS) $(SUPPRESS_FLAGS)
release: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(LIB_DIR)/LodePNG/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	
$(BUILD_DIR)/%.o: $(LIB_DIR)/%.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories
$(BUILD_DIR):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/lambda

.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  make         - Build debug executable"
	@echo "  make release - Build release executable"
	@echo "  make clean   - Remove build artifacts"

# ****************************************************************************
