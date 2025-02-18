CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O1

AVXFLAGS = -mavx512f -mfma -march=native

hw2a: CFLAGS += -pthread $(AVXFLAGS)
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp $(AVXFLAGS) -O3
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
