# ****************************************************************************
# Makefile for Lambda path tracer (courtesy Claud)
# ****************************************************************************

CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra
DEBUG_FLAGS := -g -DDEBUG -fopenmp
RELEASE_FLAGS := -g -O3 -fopenmp -Wno-maybe-uninitialized
SUPPRESS_FLAGS := -Wno-deprecated-literal-operator

SRC_DIR := src
INCLUDE_DIR := include
LIB_DIR := lib
BIN_DIR := bin
BUILD_DIR := build

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(LIB_DIR)/LodePNG/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter $(SRC_DIR)/%,$(SOURCES))) \
           $(patsubst $(LIB_DIR)/LodePNG/%.cpp,$(BUILD_DIR)/%.o,$(filter $(LIB_DIR)/LodePNG/%,$(SOURCES)))

# Include paths - find all subdirectories in include/ and lib/
INCLUDES := -I$(INCLUDE_DIR) $(addprefix -I,$(shell find $(INCLUDE_DIR) -type d))
INCLUDES += -I$(LIB_DIR) $(addprefix -I,$(shell find $(LIB_DIR) -type d))

# Target executable
TARGET := $(BIN_DIR)/lambda

# Default target (debug)
.PHONY: all debug release clean

all: debug

debug: CXXFLAGS += $(DEBUG_FLAGS) $(SUPPRESS_FLAGS)
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
