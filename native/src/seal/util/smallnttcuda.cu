#include "smallnttcuda.h"
#include <iostream>

using namespace std;

void ntt_negacyclic_harvey_lazy_(uint64_t *operand,
        const uint64_t *root_powers, const uint64_t *scaled_root_powers,
        uint64_t modulus, size_t n){
    size_t t = n >> 1;
    uint64_t *d_operand, *d_root_powers, *d_scaled_root_powers;

    cudaMalloc((void**)&d_operand, t*sizeof(uint64_t));
    cudaMalloc((void**)&d_root_powers, n*sizeof(uint64_t));
    cudaMalloc((void**)&d_scaled_root_powers, n*sizeof(uint64_t));

    cudaMemcpy(d_operand, operand,
              t*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_powers, root_powers,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_root_powers, scaled_root_powers,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    cuda_ntt_negacyclic_harvey_lazy_<<<1, 1>>>(d_operand, d_root_powers,
      d_scaled_root_powers, modulus, n);

    cudaMemcpy(operand, d_operand,
              t*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    cudaFree(d_operand);
    cudaFree(d_root_powers);
    cudaFree(d_scaled_root_powers);
}


__global__ void cuda_ntt_negacyclic_harvey_lazy_(
  uint64_t *operand,
  uint64_t *root_powers, uint64_t *scaled_root_powers,
  uint64_t modulus, size_t n
){
    uint64_t two_times_modulus = modulus * 2;

    // Return the NTT in scrambled order
    size_t t = n >> 1;
    for (size_t m = 1; m < n; m <<= 1)
    {
        if (t >= 4)
        {
            for (size_t i = 0; i < m; i++)
            {
                size_t j1 = 2 * i * t;
                size_t j2 = j1 + t;
                const uint64_t W = root_powers[m + i];
                const uint64_t Wprime = scaled_root_powers[m + i];

                uint64_t *X = operand + j1;
                uint64_t *Y = X + t;
                uint64_t currX;
                unsigned long long Q;
                for (size_t j = j1; j < j2; j += 4)
                {
                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64_(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64_(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64_(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64_(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < m; i++)
            {
                size_t j1 = 2 * i * t;
                size_t j2 = j1 + t;
                const uint64_t W = root_powers[m + i];
                const uint64_t Wprime = scaled_root_powers[m + i];

                uint64_t *X = operand + j1;
                uint64_t *Y = X + t;
                uint64_t currX;
                unsigned long long Q;
                for (size_t j = j1; j < j2; j++)
                {
                    // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in [0, 4p).
                    // X', Y' = X + WY, X - WY (mod p).
                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64_(Wprime, *Y, &Q);
                    Q = W * *Y - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);
                }
            }
        }
        t >>= 1;
    }
}

template<typename T, typename S>
__device__ inline void multiply_uint64_hw64_(
      T operand1, S operand2, unsigned long long *hw64)
        {
            auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
            auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
            operand1 >>= 32;
            operand2 >>= 32;

            auto middle1 = operand1 * operand2_coeff_right;
            T middle;
            auto left = operand1 * operand2 + (static_cast<T>(add_uint64_(
                middle1, operand2 * operand1_coeff_right, &middle)) << 32);
            auto right = operand1_coeff_right * operand2_coeff_right;
            auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

            *hw64 = static_cast<unsigned long long>(
                left + (middle >> 32) + (temp_sum >> 32));
        }

template<typename T, typename S>
__device__ inline unsigned char add_uint64_generic_(
        T operand1, S operand2, unsigned char carry,
        unsigned long long *result)
        {
            operand1 += operand2;
            *result = operand1 + carry;
            return (operand1 < operand2) || (~operand1 < carry);
        }

template<typename T, typename S>
__device__ inline unsigned char add_uint64_(
    T operand1, S operand2, unsigned char carry,
    unsigned long long *result)
{
    return add_uint64_generic_(operand1, operand2, carry, result);
}

template<typename T, typename S, typename R>
__device__ inline unsigned char add_uint64_(
    T operand1, S operand2, R *result)
{
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}
