CXX = g++
NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75

LIBS = -lboost_program_options

CSRC = kmeans.cpp arg_parser.cpp
CUDASRC = kmeans_kernel.cu

CPP_OBJS = $(CSRC:.cpp=.o)
CUDA_OBJS = $(CUDASRC:.cu=.o)

EXE = kmeans

all: $(EXE)

%.o: %.cpp
	$(NVCC) $(CFLAGS) -o $@ -c $< 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(EXE): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) -o $(EXE) $(LIBS) $(CUDA_OBJS) $(CPP_OBJS)

test:
	./$(EXE) -k 5 -d 4 -i tests/random-n2048-d16-c16.txt -m 100 -t 0.01 -c -s 42 -g -f -p


clean:
	rm -f $(EXE) $(CPP_OBJS) $(CUDA_OBJS)