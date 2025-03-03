import matplotlib.pyplot as plt
import csv
import numpy as np

def parse_e2e_file(filename):
    num_iters = 0
    times = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for line in reader:
            num_iters = int(line[0])
            times.append(num_iters * float(line[1]))

    return times

def graph_e2e(seq_times, gpu_times, shmem_times, kpp_times, k, d, n, test):
    seq_time = np.mean(seq_times)
    gpu_speedup = [seq_time / time for time in gpu_times]
    shmem_speedup = [seq_time / time for time in shmem_times]
    kpp_speedup = [seq_time / time for time in kpp_times]
    num_threads = [2 ** i for i in range(5, 11)]

    plt.figure()
    plt.plot(num_threads, gpu_speedup, marker='o', linestyle='-', color='r', label="GPU")
    plt.plot(num_threads, shmem_speedup, marker='o', linestyle='-', color='b', label="GPU Shared Memory")
    plt.plot(num_threads, kpp_speedup, marker='o', linestyle='-', color='g', label="GPU Kmeans++")
    plt.xlabel("Threads per Block")
    plt.ylabel("Speedup (T_serial / T_parallel)")
    plt.xscale("log", base=2)
    plt.xticks(num_threads, num_threads, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.title(f"Speedup comparison for n = {n}, d = {d}, k = {k}")
    plt.tight_layout()
    plt.savefig(test + "_speedup_graph")
    plt.close()

def parse_data_transfer_file(filename):
    transfer_times = []
    total_times = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for line in reader:
            transfer_times.append(float(line[0]))
            total_times.append(float(line[1]))

    return transfer_times, total_times

def graph_data_transfer(gpu_transfer_times, gpu_total_times, shmem_transfer_times, shmem_total_times, kpp_transfer_times, kpp_total_times, k, d, n, test):
    gpu_transfer = np.mean(gpu_transfer_times)
    gpu_total = np.mean(gpu_total_times)
    shmem_transfer = np.mean(shmem_transfer_times)
    shmem_total = np.mean(shmem_total_times)
    kpp_transfer = np.mean(kpp_transfer_times)
    kpp_total = np.mean(kpp_total_times)

    labels = ["GPU", "GPU Shared Memory", "GPU Kmeans++"]
    fractions = [gpu_transfer / gpu_total, shmem_transfer / shmem_total, kpp_transfer / kpp_total]

    plt.figure()
    plt.bar(labels, fractions, color=['red', 'blue', 'green'])
    plt.title(f"Data transfer time fractions for n = {n}, d = {d}, k = {k}")
    plt.xlabel("Parallel Implementations")
    plt.ylabel("Data Transfer Fraction (T_DataTransfer / T_Total)")
    plt.savefig(test + "_data_transfer_graph")
    plt.close()


small_seq_times = parse_e2e_file("e2e/small_seq.txt")
small_gpu_times = parse_e2e_file("e2e/small_gpu.txt")
small_shmem_times = parse_e2e_file("e2e/small_shmem.txt")
small_kpp_times = parse_e2e_file("e2e/small_gpu_kpp.txt")
graph_e2e(small_seq_times, small_gpu_times, small_shmem_times, small_kpp_times, 16, 16, 2048, "small")

medium_seq_times = parse_e2e_file("e2e/medium_seq.txt")
medium_gpu_times = parse_e2e_file("e2e/medium_gpu.txt")
medium_shmem_times = parse_e2e_file("e2e/medium_shmem.txt")
medium_kpp_times = parse_e2e_file("e2e/medium_gpu_kpp.txt")
graph_e2e(medium_seq_times, medium_gpu_times, medium_shmem_times, medium_kpp_times, 16, 24, 16384, "medium")

large_seq_times = parse_e2e_file("e2e/large_seq.txt")
large_gpu_times = parse_e2e_file("e2e/large_gpu.txt")
large_shmem_times = parse_e2e_file("e2e/large_shmem.txt")
large_kpp_times = parse_e2e_file("e2e/large_gpu_kpp.txt")
graph_e2e(large_seq_times, large_gpu_times, large_shmem_times, large_kpp_times, 16, 32, 65536, "large")

small_gpu_dt, small_gpu_total = parse_data_transfer_file("data_transfer/small_gpu.txt")
small_shmem_dt, small_shmem_total = parse_data_transfer_file("data_transfer/small_shmem.txt")
small_kpp_dt, small_kpp_total = parse_data_transfer_file("data_transfer/small_gpu_kpp.txt")
graph_data_transfer(small_gpu_dt, small_gpu_total, small_shmem_dt, small_shmem_total, small_kpp_dt, small_kpp_total, 16, 16, 2048, "small")

medium_gpu_dt, medium_gpu_total = parse_data_transfer_file("data_transfer/medium_gpu.txt")
medium_shmem_dt, medium_shmem_total = parse_data_transfer_file("data_transfer/medium_shmem.txt")
medium_kpp_dt, medium_kpp_total = parse_data_transfer_file("data_transfer/medium_gpu_kpp.txt")
graph_data_transfer(medium_gpu_dt, medium_gpu_total, medium_shmem_dt, medium_shmem_total, medium_kpp_dt, medium_kpp_total, 16, 24, 16384, "medium")

large_gpu_dt, large_gpu_total = parse_data_transfer_file("data_transfer/large_gpu.txt")
large_shmem_dt, large_shmem_total = parse_data_transfer_file("data_transfer/large_shmem.txt")
large_kpp_dt, large_kpp_total = parse_data_transfer_file("data_transfer/large_gpu_kpp.txt")
graph_data_transfer(large_gpu_dt, large_gpu_total, large_shmem_dt, large_shmem_total, large_kpp_dt, large_kpp_total, 16, 32, 65536, "large")