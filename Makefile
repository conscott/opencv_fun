CXX=clang++
CXXFLAGS=-g -std=c++14 -Wall -pthread $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)

BIN=facedetect

all: $(BIN)

clean: rm -f *.o
