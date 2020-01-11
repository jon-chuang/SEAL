//
// Created by jonch on 08/01/2020.
//

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <assert.h>

extern void ntt_negacyclic_harvey_lazy_(std::uint64_t *operand,
        const std::uint64_t *root_powers,
        const std::uint64_t *scaled_root_powers,
        std::uint64_t modulus, size_t n);

extern void inverse_ntt_negacyclic_harvey_lazy_(uint64_t *operand,
        const uint64_t *inv_root_powers_div_two,
        const uint64_t *scaled_inv_root_powers_div_two,
        uint64_t modulus, size_t n);



__global__ void cuda_ntt_negacyclic_harvey_lazy(
  std::uint64_t *operand,
  const std::uint64_t * __restrict__ root_powers,
  const std::uint64_t * __restrict__ scaled_root_powers,
  std::uint64_t modulus, size_t n
);

__device__ void cuda_ntt_negacyclic_harvey_lazy_(
  std::uint64_t *operand,
  const std::uint64_t * __restrict__ root_powers,
  const std::uint64_t * __restrict__ scaled_root_powers,
  std::uint64_t modulus, size_t n
);

__global__ void cuda_ntt_negacyclic_harvey(
  uint64_t *operand,
  const uint64_t * __restrict__ root_powers,
  const uint64_t * __restrict__ scaled_root_powers,
  uint64_t modulus, size_t n
);

__device__ void cuda_ntt_negacyclic_harvey_(
  uint64_t *operand,
  const uint64_t * __restrict__ root_powers,
  const uint64_t * __restrict__ scaled_root_powers,
  uint64_t modulus, size_t n
);



__global__ void cuda_inverse_ntt_negacyclic_harvey_lazy(
    uint64_t *operand,
    const uint64_t * __restrict__ inv_root_powers_div_two,
    const uint64_t * __restrict__ scaled_inv_root_powers_div_two,
    uint64_t modulus, size_t n
);

__device__ void cuda_inverse_ntt_negacyclic_harvey_lazy_(
    uint64_t *operand,
    const uint64_t * __restrict__ inv_root_powers_div_two,
    const uint64_t * __restrict__ scaled_inv_root_powers_div_two,
    uint64_t modulus, size_t n
);

__global__ void cuda_inverse_ntt_negacyclic_harvey(
    uint64_t *operand,
    const uint64_t * __restrict__ inv_root_powers_div_two,
    const uint64_t * __restrict__ scaled_inv_root_powers_div_two,
    uint64_t modulus, size_t n
);

__device__ void cuda_inverse_ntt_negacyclic_harvey_(
    uint64_t *operand,
    const uint64_t * __restrict__ inv_root_powers_div_two,
    const uint64_t * __restrict__ scaled_inv_root_powers_div_two,
    uint64_t modulus, size_t n
);

__global__ void cuda_ntt_negacyclic_harvey_lazy_v3(
  std::uint64_t *operand,
  const std::uint64_t * __restrict__ root_powers,
  const std::uint64_t * __restrict__ scaled_root_powers,
  std::uint64_t modulus, size_t n
);

template<typename T, typename S>
__device__ inline void multiply_uint64_hw64_(
      T operand1, S operand2, unsigned long long *hw64);

template<typename T, typename S>
__device__ inline unsigned char add_uint64_generic_(
        T operand1, S operand2, unsigned char carry,
        unsigned long long *result);

template<typename T, typename S>
__device__ inline unsigned char add_uint64_(
    T operand1, S operand2, unsigned char carry,
    unsigned long long *result);

template<typename T, typename S, typename R>
__device__ inline unsigned char add_uint64_(
    T operand1, S operand2, R *result);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T> __device__ void inline swap_test_device(T& a, T& b)
{
    T c(a); a=b; b=c;
}
