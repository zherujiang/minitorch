import minitorch
import time
import numpy as np

import matplotlib.pyplot as plt
from minitorch.tensor import Tensor

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> Tensor:
    """Run a matrix multiplication with the given backend."""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y
    return z


def draw_plot(times: dict[int, dict[str, float]]) -> None:
    """Draw a plot of the runtime of the different backends for different matrix sizes."""
    sizes = list(times.keys())
    graph = dict()

    # convert times dictionary to list for plotting
    for size in sizes:
        for backend, runtime in times[size].items():
            if backend not in graph:
                graph[backend] = []
            graph[backend].append(runtime)

    # draw plot
    plt.figure(figsize=(10, 6))
    for backend in graph.keys():
        plt.plot(sizes, graph[backend], label=backend)
    plt.title("Matrix Multiplication Speed Test")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Training Time summary (s)")
    for size, stimes in times.items():
        print(f"Matrix Size: {size}")
        for backend, runtime in stimes.items():
            print(f"    {backend}: {runtime:.5f}")

    draw_plot(times)
