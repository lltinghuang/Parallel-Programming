CXX = mpicxx
LDLIBS = -pthread
CFLAGS = -O3 -fopenmp -msse2
CXXFLAGS = -O3 -fopenmp -msse2

NVCC = nvcc
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
CUDA_PATH = /opt/cuda-12.6
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
LDFLAGS = -lm

TARGETS = hw3-1 hw3-2

.PHONY: all clean hw3-1 hw3-2

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(TARGETS:=.o)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CUDA_LIBS)

hw3-2-amd: hw3-2.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<
