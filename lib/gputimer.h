// gputimer.h

#ifndef __GPUTIMER_H__
#define __GPUTIMER_H__

#include <cuda.h>
#include <cuda_runtime.h>

class GpuTimer {
   public:
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return elapsedTime;
    }

   private:
    cudaEvent_t start;
    cudaEvent_t stop;
};

#endif  // __GPUTIMER_H__
