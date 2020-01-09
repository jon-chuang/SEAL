#include <iostream>
#include <math.h>
#include "test.h"

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

__global__ void build_table(){
  int i;
}

namespace seal
{
  int cuda_host(void)
  {
    int N = 1<<25;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    int entry = -1;
    for (int i = 0; i < N; i++)
      if (maxError < fabs(y[i]-3.0f)){
        maxError = fabs(y[i]-3.0f);
        entry = i;
      }
    std::cout << "Max error: " << maxError << " at entry " << entry << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
  }

  int cuda_unified_mem(void)
  {
    int i;
  }
}
