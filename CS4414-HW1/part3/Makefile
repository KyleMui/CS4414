# Makefile for building knn_arglib

CXX = g++
CXXFLAGS = -std=c++11 -Ialglib-cpp/src
SRC = main.cpp alglib-cpp/src/*.cpp
TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)