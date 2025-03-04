NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

LIBS = -lboost_program_options -lcudart

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

	# END TO END / DATA TRANSFER TESTING

	# bash -c 'num=32; while [[ $$num -le 1024 ]]; do \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -z $$num >> performance/data_transfer/small_seq.txt; \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -g -z $$num >> performance/data_transfer/small_gpu.txt; \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -f -z $$num >> performance/data_transfer/small_shmem.txt; \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -p -z $$num >> performance/data_transfer/small_seq_kpp.txt; \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -g -p -z $$num >> performance/data_transfer/small_gpu_kpp.txt; \
	# 	./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -f -p -z $$num >> performance/data_transfer/small_shmem_kpp.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -z $$num >> performance/data_transfer/medium_seq.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -g -z $$num >> performance/data_transfer/medium_gpu.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -f -z $$num >> performance/data_transfer/medium_shmem.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -p -z $$num >> performance/data_transfer/medium_seq_kpp.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -g -p -z $$num >> performance/data_transfer/medium_gpu_kpp.txt; \
	# 	./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -f -p -z $$num >> performance/data_transfer/medium_shmem_kpp.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -z $$num >> performance/data_transfer/large_seq.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -g -z $$num >> performance/data_transfer/large_gpu.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -f -z $$num >> performance/data_transfer/large_shmem.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -p -z $$num >> performance/data_transfer/large_seq_kpp.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -g -p -z $$num >> performance/data_transfer/large_gpu_kpp.txt; \
	# 	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -f -p -z $$num >> performance/data_transfer/large_shmem_kpp.txt; \
	# 	((num *= 2)); \
	# done'

	# ./$(EXE) -k 16 -d 16 -i tests/random-n2048-d16-c16.txt -m 10000 -t 0.0001 -c -p -z 1024
	# ./$(EXE) -k 16 -d 24 -i tests/random-n16384-d24-c16.txt -m 10000 -t 0.0001 -c -p -z 1024
	./$(EXE) -k 16 -d 32 -i tests/random-n65536-d32-c16.txt -m 10000 -t 0.0001 -c -p -z 1024


clean:
	rm -f $(EXE) $(CPP_OBJS) $(CUDA_OBJS)