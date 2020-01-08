//
// Created by jonch on 08/01/2020.
//

#pragma once
#include <cuda_runtime.h>
// #include "seal/util/smallntt.h"

using namespace std;
using namespace seal;

__global__ void cuda_ntt_negacyclic_harvey_lazy(uint64_t *operand,
                                                const util::SmallNTTTables &tables, uint64_t modulus);

__device__ inline void multiply_uint64_hw64(
    uint64_t operand1, uint64_t operand2, unsigned long long *hw64);


__device__ inline unsigned char add_uint64_generic(
            uint64_t operand1, uint64_t operand2, unsigned char carry,
            unsigned long long *result);

__device__ inline unsigned char add_uint64(
    uint64_t operand1, uint64_t operand2, unsigned char carry,
    unsigned long long *result);

__device__ inline unsigned char add_uint64(
    uint64_t operand1, uint64_t operand2, unsigned char *result);
