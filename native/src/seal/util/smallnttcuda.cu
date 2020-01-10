#include "smallnttcuda.cuh"
#include <iostream>
#include <cooperative_groups.h>
#include <assert.h>
#include <stdio.h>

using namespace std;
namespace cg = cooperative_groups;

void ntt_negacyclic_harvey_lazy_(uint64_t *operand,
        const uint64_t *root_powers, const uint64_t *scaled_root_powers,
        uint64_t modulus, size_t n){
    uint64_t *d_operand, *d_root_powers, *d_scaled_root_powers;

    cudaMalloc((void**)&d_operand, n*sizeof(uint64_t));
    cudaMalloc((void**)&d_root_powers, n*sizeof(uint64_t));
    cudaMalloc((void**)&d_scaled_root_powers, n*sizeof(uint64_t));

    cudaMemcpy(d_operand, operand,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_powers, root_powers,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_root_powers, scaled_root_powers,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    size_t blocksize = min(n/2, size_t(1024));

    dim3 block_dim(blocksize, 1, 1);
    dim3 grid_dim(n/(2*blocksize), 1, 1);

    // printf("Launching Kernel %d %d %d\n", n, grid_dim.x, block_dim.x);
    cuda_ntt_negacyclic_harvey_lazy_<<<grid_dim, block_dim, 2*blocksize*sizeof(uint64_t)>>>
        (d_operand, d_root_powers,d_scaled_root_powers, modulus, n);

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(operand, d_operand,
              n*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    // printf("Kernel completed Success\n");
    cudaFree(d_operand);
    cudaFree(d_root_powers);
    cudaFree(d_scaled_root_powers);
}

void inverse_ntt_negacyclic_harvey_lazy_(uint64_t *operand,
        const uint64_t *inv_root_powers_div_two,
        const uint64_t *scaled_inv_root_powers_div_two,
        uint64_t modulus, size_t n)
{
    uint64_t *d_operand, *d_inv_root_powers_div_two, *d_scaled_inv_root_powers_div_two;

    cudaMalloc((void**)&d_operand, n*sizeof(uint64_t));
    cudaMalloc((void**)&d_inv_root_powers_div_two, n*sizeof(uint64_t));
    cudaMalloc((void**)&d_scaled_inv_root_powers_div_two, n*sizeof(uint64_t));

    cudaMemcpy(d_operand, operand,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_root_powers_div_two, inv_root_powers_div_two,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_inv_root_powers_div_two, scaled_inv_root_powers_div_two,
              n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    size_t blocksize = min(n/2, size_t(1024));

    dim3 block_dim(blocksize, 1, 1);
    dim3 grid_dim(n/(2*blocksize), 1, 1);

    // printf("Launching Kernel %d %d %d\n", n, grid_dim.x, block_dim.x);
    cuda_inverse_ntt_negacyclic_harvey_lazy_<<<grid_dim, block_dim, 2*blocksize*sizeof(uint64_t)>>>
        (d_operand, d_inv_root_powers_div_two,d_scaled_inv_root_powers_div_two, modulus, n);

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(operand, d_operand,
              n*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    // printf("Kernel completed Success\n");
    cudaFree(d_operand);
    cudaFree(d_inv_root_powers_div_two);
    cudaFree(d_scaled_inv_root_powers_div_two);
}

// We divide butterfly into three phases: One where the
// partition size is larger than 1024, one where the partition size
// is smaller or equal to 64, and the everything else that sits in the middle.
// In the cross-block phase, we read data from global memory and loop over
// the lanes by means of a grid stride loop.
//
// In phases two and three, we exploit embarassing parallelism
// In phase 2, we transfer data to shared memory, and perform
// exchanges block-wise
//
// Finally, in the warp-size partition phase, we use __shfl_xor_sync
// to perform a within-warp butterfly.

__global__ void cuda_ntt_negacyclic_harvey_lazy_(
  uint64_t *operand,
  const uint64_t * __restrict__ root_powers,
  const uint64_t * __restrict__ scaled_root_powers,
  uint64_t modulus, size_t n
){
    // printf("%d\n", n);
    extern __shared__ uint64_t shared_operand[];
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto grid = cg::this_grid();

    uint64_t two_times_modulus = modulus * 2;
    size_t t = n >> 1;

    for (size_t m = 1; m < n; m <<= 1)
    {
      // Loop over gridIdx.x
        for (size_t k = 0;
          2*(k*gridDim.x*blockDim.x + tid) < n; k++)
        {
            size_t i = (k*gridDim.x*blockDim.x + tid) / t; // partition number
            size_t local_id = (k*gridDim.x*blockDim.x + tid) % t;
            size_t j1 = i * 2 * t; // offset
            const uint64_t W = root_powers[m + i];
            const uint64_t Wprime = scaled_root_powers[m + i];

            uint64_t currX;
            unsigned long long Q;

            uint64_t X = operand[j1+local_id];
            uint64_t Y = operand[j1+local_id+t];

            currX = X - (two_times_modulus & static_cast<uint64_t>(
                          -static_cast<int64_t>(X >= two_times_modulus)));
            multiply_uint64_hw64_(Wprime, Y, &Q);
            Q = Y * W - Q * modulus;
            operand[j1+local_id] = currX + Q;
            operand[j1+local_id+t] = currX + (two_times_modulus - Q);
        }
        t >>= 1;
        grid.sync();
        if (t == blockDim.x) { //load from global memory
            shared_operand[threadIdx.x] = operand[tid];
        }
    } if(false) {
        printf("unreachable\n");
    // // Loop over gridDim
    // for (size_t offset = tid /(two_times_t);
    //       offset < (n >> 1); i += gridDim.x*blockDim.x)
    // {
    //   for (size_t m = dimBlock.x; m < n; m <<= 1){
    //       if (t > 32) {
    //         for (size_t i = 0; i < m; i++)
    //         {
    //             size_t j1 = 2 * i * t;
    //             size_t j2 = j1 + t;
    //             const uint64_t W = root_powers[m + i];
    //             const uint64_t Wprime = scaled_root_powers[m + i];
    //
    //             uint64_t currX;
    //             unsigned long long Q;
    //
    //             for (size_t j = tid % two_times_t; j < t;
    //                   j += gridDim.x*)
    //             {
    //                 uint64_t X = shared_operand[j];
    //                 uint64_t Y = shared_operand[j+t];
    //
    //                 currX = X - (two_times_modulus & static_cast<uint64_t>(
    //                               -static_cast<int64_t>(X >= two_times_modulus)));
    //                 multiply_uint64_hw64_(Wprime, Y, &Q);
    //                 Q = Y * W - Q * modulus;
    //                 shared_operand[j] = currX + Q;
    //                 shared_operand[j+t] = currX + (two_times_modulus - Q);
    //             }
    //         }
    //         __syncthreads();
    //     } else {
    //         uint64_t X = shared_operand[tid % blockDim.x];
    //         uint64_t Y = shared_operand[+t];
    //
    //         for (int i=16; i>=1; i/=2){
    //             if (threadIdx.x & i == 0) swap_test_device(X, Y);
    //             Y = __shfl_xor_sync(0xFFFFFFFF, X, i);
    //             if (threadIdx.x & i == 0) swap_test_device(X, Y);
    //
    //             currX = X - (two_times_modulus & static_cast<uint64_t>(
    //                           -static_cast<int64_t>(X >= two_times_modulus)));
    //             multiply_uint64_hw64_(Wprime, Y, &Q);
    //             Q = Y * W - Q * modulus;
    //             X = currX + Q;
    //             Y = currX + (two_times_modulus - Q);
    //         }
    //         for (k=2*, k<, k += 2*)
    //         shared_operand[] = X;
    //         shared_operand[+1] = Y;
    //     }
    //     t >>= 1;
    //     two_times_t >>= 1;
    //     }
    //     shared_operand[threadIdx.x] = operand[tid]
    //     operand[] = shared_operand[];
    // }
    }
}

__global__ void cuda_inverse_ntt_negacyclic_harvey_lazy_(
    uint64_t *operand,
    const uint64_t * __restrict__ inv_root_powers_div_two,
    const uint64_t * __restrict__ scaled_inv_root_powers_div_two,
    uint64_t modulus, size_t n
){
    extern __shared__ uint64_t shared_operand[];
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto grid = cg::this_grid();

    uint64_t two_times_modulus = modulus * 2;
    size_t t = 1;

    for (size_t m = n; m > 1; m >>= 1)
    {
        size_t h = m >> 1;

        for (size_t k = 0;
          2*(k*gridDim.x*blockDim.x + tid) < n; k++)
        {
            size_t i = (k*gridDim.x*blockDim.x + tid) / t; // partition number
            size_t local_id = (k*gridDim.x*blockDim.x + tid) % t;
            // Need the powers of phi^{-1} in bit-reversed order
            size_t j1 = i * 2 * t; // offset
            const uint64_t W = inv_root_powers_div_two[h + i];
            const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

            uint64_t U = operand[j1+local_id];
            uint64_t V = operand[j1+local_id+t];
            uint64_t currU;
            uint64_t T;
            unsigned long long H;

            T = two_times_modulus - V + U;
            currU = U + V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((U << 1) >= T)));
            U = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
            multiply_uint64_hw64_(Wprime, T, &H);
            V = T * W - H * modulus;
            operand[j1+local_id] = U;
            operand[j1+local_id+t] = V;
        }
        t <<= 1;
    }
}

// In this rendition of NTT, we try to load the entire problem
// into shared memory

// Original implementation
__global__ void cuda_ntt_negacyclic_harvey_lazy_v3(
  uint64_t *operand,
  const uint64_t * __restrict__ root_powers,
  const uint64_t * __restrict__ scaled_root_powers,
  uint64_t modulus, size_t n
){
    uint64_t two_times_modulus = modulus * 2;

    // Return the NTT in scrambled order
    size_t t = n >> 1;
    for (size_t m = 1; m < n; m <<= 1)
    {

        if (t >= 32)
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

// int main()
// {
//     MemoryPoolHandle pool = MemoryPoolHandle::Global();
//     SmallNTTTables tables;
//
//     int coeff_count_power = 3;
//     SmallModulus modulus(0xffffffffffc0001ULL);
//     auto poly(allocate_zero_poly(800, 1, pool));
//     auto temp(allocate_zero_poly(800, 1, pool));
//     tables.generate(coeff_count_power, modulus)
//
//     inverse_ntt_negacyclic_harvey(poly.get(), tables);
//
//     random_device rd;
//     for (size_t i = 0; i < 800; i++)
//     {
//         poly[i] = static_cast<uint64_t>(rd()) % modulus.value();
//         temp[i] = poly[i];
//     }
//
//     size_t n = size_t(1) << tables.coeff_count_power();
//     const uint64_t *root_powers = tables.get_root_powers();
//     const uint64_t *scaled_root_powers = tables.get_scaled_root_powers();
//     uint64_t modulus = tables.modulus().value();
//     ntt_negacyclic_harvey_lazy_(poly.get(), root_powers,
//       scaled_root_powers, modulus, n);
//
//     inverse_ntt_negacyclic_harvey(poly.get(), tables);
//
//    return 0;
// }
