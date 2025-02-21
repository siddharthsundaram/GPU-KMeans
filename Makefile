CXX = g++
NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC

LIBS = -lboost_program_options

CSRC = arg_parser.cpp
CUDASRC = kmeans.cu

CPP_OBJS = $(CSRC:.cpp=.o)
CUDA_OBJS = $(CUDASRC:.cu=.o)

EXE = kmeans

all: $(EXE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXE): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(CUDA_OBJS) $(CPP_OBJS) -o $(EXE) $(LIBS)

test:
	# ./$(EXE) -k 5 -d 3 -i data.txt -m 100 -t 0.01
	./$(EXE) -k 5 -d 3 -i data.txt -m 100 -t 0.01 -c -s 42 -g -f -p


clean:
	rm -f $(EXE) $(CPP_OBJS) $(CUDA_OBJS)