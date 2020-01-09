//
// Created by jonch on 08/01/2020.
//

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

extern void ntt_negacyclic_harvey_lazy_(std::uint64_t *operand,
        const std::uint64_t *root_powers, const std::uint64_t *scaled_root_powers,
        std::uint64_t modulus, size_t n);

__global__ void cuda_ntt_negacyclic_harvey_lazy_(
  std::uint64_t *operand,
  std::uint64_t* root_powers, std::uint64_t* scaled_root_powers,
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
