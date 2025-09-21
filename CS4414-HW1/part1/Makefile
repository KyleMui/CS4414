# Makefile for compiling main.cpp with knn.hpp

CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall
TARGET = main
SRCS = main.cpp knn.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp knn.hpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
