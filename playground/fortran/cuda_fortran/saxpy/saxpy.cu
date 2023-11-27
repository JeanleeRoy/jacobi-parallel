// compile for library: nvcc -arch=sm_35 -Xcompiler -fPIC -shared -o libcuda_saxpy.so cuda_saxpy.cu
// simple compile: nvcc -c saxpy.cu

__global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" void cuda_saxpy(float a, float *x, float *y, int n) {
    float *d_x, *d_y;

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int tblock = 512;
    int grid = (n + tblock - 1) / tblock;

    saxpy<<<grid, tblock>>>(a, d_x, d_y, n);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
}
