#include "seal/util/smallnttcuda.h"
#include "seal/util/smallntt.h"
#include <stdint.h>


void ntt_negacyclic_harvey_lazy_(uint64_t *operand,
        const uint64_t *root_powers, const uint64_t *scaled_root_powers,
        uint64_t modulus, size_t n){
    size_t t = n >> 1;
    uint64_t *d_operand;
    cudaMallocManaged(&d_operand, t*sizeof(uint64_t));
    d_operand = operand;  // Use some form of zerocopy semantics?

    cuda_ntt_negacyclic_harvey_lazy_<<<1, 1>>>(operand, root_powers,
      scaled_root_powers, modulus, n);

    cudaDeviceSynchronize();
    cudaFree(d_operand);
    // *operand = *d_operand;
}


__global__ void cuda_ntt_negacyclic_harvey_lazy_(
  uint64_t *operand,
  const uint64_t *root_powers, const uint64_t *scaled_root_powers,
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
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
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
                    multiply_uint64_hw64(Wprime, *Y, &Q);
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
__device__ inline void multiply_uint64_hw64(
      T operand1, S operand2, unsigned long long *hw64)
        {
            auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
            auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
            operand1 >>= 32;
            operand2 >>= 32;

            auto middle1 = operand1 * operand2_coeff_right;
            T middle;
            auto left = operand1 * operand2 + (static_cast<T>(add_uint64(
                middle1, operand2 * operand1_coeff_right, &middle)) << 32);
            auto right = operand1_coeff_right * operand2_coeff_right;
            auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

            *hw64 = static_cast<unsigned long long>(
                left + (middle >> 32) + (temp_sum >> 32));
        }

template<typename T, typename S>
__device__ inline unsigned char add_uint64_generic(
        T operand1, S operand2, unsigned char carry,
        unsigned long long *result)
        {
            operand1 += operand2;
            *result = operand1 + carry;
            return (operand1 < operand2) || (~operand1 < carry);
        }

template<typename T, typename S>
__device__ inline unsigned char add_uint64(
    T operand1, S operand2, unsigned char carry,
    unsigned long long *result)
{
    return add_uint64_generic(operand1, operand2, carry, result);
}

template<typename T, typename S, typename R>
__device__ inline unsigned char add_uint64(
    T operand1, S operand2, R *result)
{
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}
