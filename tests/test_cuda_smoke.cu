#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

__global__ void write_one(int *output) {
    output[0] = 1;
}

int main() {
    int *device_output = nullptr;
    int host_output = 0;

    cudaError_t status = cudaMalloc(&device_output, sizeof(int));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(status)
                  << "\n";
        return EXIT_FAILURE;
    }

    write_one<<<1, 1>>>(device_output);

    status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "kernel launch failed: " << cudaGetErrorString(status)
                  << "\n";
        cudaFree(device_output);
        return EXIT_FAILURE;
    }

    status = cudaMemcpy(&host_output, device_output, sizeof(int),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(status)
                  << "\n";
        cudaFree(device_output);
        return EXIT_FAILURE;
    }

    cudaFree(device_output);

    if (host_output != 1) {
        std::cerr << "expected 1, got " << host_output << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[PASS] cuda smoke\n";
    return EXIT_SUCCESS;
}