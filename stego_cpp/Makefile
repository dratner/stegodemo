# Makefile for StegoGPT C++ implementation

CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -I./include
LDFLAGS := -pthread

# Check if we're on macOS and need to use clang
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXX := clang++
endif

# Source files
SRCS := src/stego.cpp src/main.cpp src/gpt2_model.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := stego_app

.PHONY: all clean

all: $(TARGET)

# Build our app
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)