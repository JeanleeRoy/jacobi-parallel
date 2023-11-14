#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device: %s\n", props.name);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);

    printf("Max block size: %d x %d x %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);

    printf("Max grid size: %d x %d x %d\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

    // double local_memory = (double)props.totalConstMem / 1024;
    // printf("Local memory per thread: %2.f KB\n", local_memory);
    
    double shared_memory = (double)props.sharedMemPerBlock /  1024;
    printf("Shared memory per block: %2.f KB\n", shared_memory);

    double total_gpu_memory = (double)props.totalGlobalMem / (1024 * 1024 * 1024);
    printf("Total GPU memory: %2.f GB\n", total_gpu_memory);

    return 0;
}

// Compile: nvcc -o gpu_info gpu_info.cu

// More info: $ nvidia-smi
