// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/smallntt.h"
#include "seal/util/polyarith.h"
#include "seal/util/uintarith.h"
#include "seal/smallmodulus.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/defines.h"
#include <CL/sycl.hpp>
#include <algorithm>

using namespace std;
using namespace cl;

// Create a .cu version to replace the entire functionality
// of ntt/intt and tables with minimal data movement
// Future work: replace entire memory management strategy w/ unified memory (?)

namespace seal
{
    namespace util
    {
      SmallNTTTables::SmallNTTTables(int coeff_count_power,
          const SmallModulus &modulus, MemoryPoolHandle pool) :
          pool_(move(pool))
      {
#ifdef SEAL_DEBUG
          if (!pool_)
          {
              throw invalid_argument("pool is uninitialized");
          }
#endif
          if (!generate(coeff_count_power, modulus))
          {
              // Generation failed; probably modulus wasn't prime.
              // It is necessary to check generated() after creating
              // this class.
          }
      }

      void SmallNTTTables::reset()
      {
          generated_ = false;
          modulus_ = SmallModulus();
          root_ = 0;
          root_powers_.release();
          scaled_root_powers_.release();
          inv_root_powers_.release();
          scaled_inv_root_powers_.release();
          inv_root_powers_div_two_.release();
          scaled_inv_root_powers_div_two_.release();
          inv_degree_modulo_ = 0;
          coeff_count_power_ = 0;
          coeff_count_ = 0;
      }

      bool SmallNTTTables::generate(int coeff_count_power,
          const SmallModulus &modulus)
      {
          reset();

          if ((coeff_count_power < get_power_of_two(SEAL_POLY_MOD_DEGREE_MIN)) ||
              coeff_count_power > get_power_of_two(SEAL_POLY_MOD_DEGREE_MAX))
          {
              throw invalid_argument("coeff_count_power out of range");
          }

          coeff_count_power_ = coeff_count_power;
          coeff_count_ = size_t(1) << coeff_count_power_;

          // Allocate memory for the tables
          root_powers_ = allocate_uint(coeff_count_, pool_);
          inv_root_powers_ = allocate_uint(coeff_count_, pool_);
          scaled_root_powers_ = allocate_uint(coeff_count_, pool_);
          scaled_inv_root_powers_ = allocate_uint(coeff_count_, pool_);
          inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
          scaled_inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
          modulus_ = modulus;

          // We defer parameter checking to try_minimal_primitive_root(...)
          if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
          {
              reset();
              return false;
          }

          uint64_t inverse_root;
          if (!try_invert_uint_mod(root_, modulus_, inverse_root))
          {
              reset();
              return false;
          }

          // Populate the tables storing (scaled version of) powers of root
          // mod q in bit-scrambled order.
          ntt_powers_of_primitive_root(root_, root_powers_.get());
          ntt_scale_powers_of_primitive_root(root_powers_.get(),
              scaled_root_powers_.get());

          // Populate the tables storing (scaled version of) powers of
          // (root)^{-1} mod q in bit-scrambled order.
          ntt_powers_of_primitive_root(inverse_root, inv_root_powers_.get());
          ntt_scale_powers_of_primitive_root(inv_root_powers_.get(),
              scaled_inv_root_powers_.get());

          // Populate the tables storing (scaled version of ) 2 times
          // powers of roots^-1 mod q  in bit-scrambled order.
          for (size_t i = 0; i < coeff_count_; i++)
          {
              inv_root_powers_div_two_[i] =
                  div2_uint_mod(inv_root_powers_[i], modulus_);
          }
          ntt_scale_powers_of_primitive_root(inv_root_powers_div_two_.get(),
              scaled_inv_root_powers_div_two_.get());

          // Last compute n^(-1) modulo q.
          uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
          generated_ = try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_);

          if (!generated_)
          {
              reset();
              return false;
          }
          return true;
      }

        void SmallNTTTables::ntt_powers_of_primitive_root(uint64_t root,
            uint64_t *destination) const
        {
            uint64_t *destination_start = destination;
            *destination_start = 1;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                uint64_t *next_destination =
                    destination_start + reverse_bits(i, coeff_count_power_);
                *next_destination =
                    multiply_uint_uint_mod(*destination, root, modulus_);
                destination = next_destination;
            }
        }

        // compute floor ( input * beta /q ), where beta is a 64k power of 2
        // and  0 < q < beta.
        void SmallNTTTables::ntt_scale_powers_of_primitive_root(
            const uint64_t *input, uint64_t *destination) const
        {
            for (size_t i = 0; i < coeff_count_; i++, input++, destination++)
            {
                uint64_t wide_quotient[2]{ 0, 0 };
                uint64_t wide_coeff[2]{ 0, *input };
                divide_uint128_uint64_inplace(wide_coeff, modulus_.value(), wide_quotient);
                *destination = wide_quotient[0];
            }
        }

        /**
        This function computes in-place the negacyclic NTT. The input is
        a polynomial a of degree n in R_q, where n is assumed to be a power of
        2 and q is a prime such that q = 1 (mod 2n).

        The output is a vector A such that the following hold:
        A[j] =  a(psi**(2*bit_reverse(j) + 1)), 0 <= j < n.

        For details, see Michael Naehrig and Patrick Longa.
        */

        // For testing purposes
        void ntt_negacyclic_harvey_lazy(uint64_t *operand, const SmallNTTTables &tables)
        {
            size_t n = size_t(1) << tables.coeff_count_power();
            const uint64_t *root_powers = tables.get_root_powers();
            const uint64_t *scaled_root_powers = tables.get_scaled_root_powers();
            uint64_t modulus = tables.modulus().value();
            // sycl::queue q;
            // sycl::buffer<uint64_t> buf_rp(root_powers, n);
            // sycl::buffer<uint64_t> buf_srp(scaled_root_powers, n);
            // sycl::buffer<uint64_t> buf_operand(operand, n);
            //
            // ntt_negacyclic_harvey_(q, buf_operand, buf_rp, buf_srp, modulus, n, true);

            ntt_negacyclic_harvey_lazy__(operand, root_powers, scaled_root_powers, modulus, n);
        }


        void ntt_negacyclic_harvey_(
          sycl::queue& q,
          sycl::buffer<uint64_t> buf_operand,
          sycl::buffer<uint64_t>& buf_rp,
          sycl::buffer<uint64_t>& buf_srp,
          uint64_t modulus, size_t n, bool lazy
        ){
            sycl::nd_range<1> work_items{sycl::range<1>
                (1024*((n/2+1023)/(1024))), sycl::range<1>(1024)};

            q.submit([&](sycl::handler& cgh){
            auto _rp = buf_rp.get_access<sycl::access::mode::read>(cgh);
            auto _srp = buf_srp.get_access<sycl::access::mode::read>(cgh);
            auto _operand = buf_operand.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class _ntt_negacyclic_harvey>
                (work_items, [=](sycl::nd_item<1> it){
                    int tid = it.get_group(0) * work_items.get_local_range().get(0) + it.get_local_id(0);
                    if (2*tid < n){

                    uint64_t two_times_modulus = modulus * 2;
                    size_t t = n >> 1;

                    for (size_t m = 1; m < n; m <<= 1)
                    {
                        size_t i = tid / t; // partition number
                        size_t local_id = tid % t;
                        size_t j1 = i * 2 * t; // offset
                        const uint64_t W = _rp[m + i];
                        const uint64_t Wprime = _srp[m + i];

                        uint64_t currX;
                        unsigned long long Q;

                        uint64_t X = _operand[j1+local_id];
                        uint64_t Y = _operand[j1+local_id+t];

                        currX = X - (two_times_modulus & static_cast<uint64_t>(
                                      -static_cast<int64_t>(X >= two_times_modulus)));
                        multiply_uint64_hw64(Wprime, Y, &Q);
                        Q = Y * W - Q * modulus;
                        _operand[j1+local_id] = currX + Q;
                        _operand[j1+local_id+t] = currX + (two_times_modulus - Q);

                        t >>= 1;
                        it.barrier();
                    }
                    if (!lazy){
                      if (_operand[tid*2] >= two_times_modulus) _operand[tid*2] -= two_times_modulus;
                      if (_operand[tid*2] >= modulus) _operand[tid*2] -= modulus;

                      if (_operand[tid*2+1] >= two_times_modulus) _operand[tid*2+1] -= two_times_modulus;
                      if (_operand[tid*2+1] >= modulus) _operand[tid*2+1] -= modulus;
                    }
                  }
              });
          });
      }

      void ntt_negacyclic_harvey_lazy__(
        uint64_t *operand,
        const uint64_t *root_powers, const uint64_t *scaled_root_powers,
        uint64_t modulus, size_t n
      ){
          uint64_t two_times_modulus = modulus * 2;

          // Return the NTT in scrambled order
          size_t t = n >> 1;
          for (size_t m = 1; m < n; m <<= 1)
          {
              for (size_t tid = 0; 2*tid < n; tid++)
              {
                  size_t i = tid / t;
                  size_t local_id = tid % t;
                  size_t j1 = 2 * i * t;
                  const uint64_t W = root_powers[m + i];
                  const uint64_t Wprime = scaled_root_powers[m + i];

                  uint64_t currX;
                  unsigned long long Q;

                  uint64_t X = operand[j1+local_id];
                  uint64_t Y = operand[j1+local_id+t];

                  // currX = X - (two_times_modulus & static_cast<uint64_t>(
                  //               -static_cast<int64_t>(X >= two_times_modulus)));
                  // multiply_uint64_hw64(Wprime, Y, &Q);
                  // Q = Y * W - Q * modulus;
                  // operand[j1+local_id] = currX + Q;
                  // operand[j1+local_id+t] = currX + (two_times_modulus - Q);

                  currX = operand[j1+local_id] - (two_times_modulus & static_cast<uint64_t>
                      (-static_cast<int64_t>(operand[j1+local_id] >= two_times_modulus)));
                  multiply_uint64_hw64(Wprime, operand[j1+local_id+t], &Q);
                  Q = operand[j1+local_id+t] * W - Q * modulus;
                  operand[j1+local_id] = currX + Q;
                  operand[j1+local_id+t] = currX + (two_times_modulus - Q);
              }
              t >>= 1;
          }
      }


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

        void ntt_negacyclic_harvey_lazy___(
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

        // Wrapper to original interface
        void inverse_ntt_negacyclic_harvey_lazy(uint64_t *operand, const SmallNTTTables &tables)
        {
            size_t n = size_t(1) << tables.coeff_count_power();
            const size_t n_ = (const size_t) n;
            const uint64_t *inv_root_powers_div_two = tables.get_inv_root_powers_div_two();
            const uint64_t *scaled_inv_root_powers_div_two = tables.get_scaled_inv_root_powers_div_two();
            uint64_t modulus = tables.modulus().value();

            vector<uint64_t> operand_;
            operand_.assign(operand, operand+n);

            sycl::buffer<uint64_t> buf_irp(inv_root_powers_div_two, n);
            sycl::buffer<uint64_t> buf_sirp(scaled_inv_root_powers_div_two, n);
            sycl::buffer<uint64_t> buf_operand(operand_.data(), n);
            sycl::queue q;

            inverse_ntt_negacyclic_harvey_(q, buf_operand, buf_irp, buf_sirp, modulus, n, true);

            inverse_ntt_negacyclic_harvey_lazy__(operand, inv_root_powers_div_two, scaled_inv_root_powers_div_two, modulus, n);

            buf_operand.get_access<sycl::access::mode::read_write>();
            for (size_t i = 0; i < n; i++){
                if (operand_[i] != operand[i]) {
                  cout << operand_[i] << " " << operand[i] << endl;
                  operand_[i] = operand[i];
                }
            }
        }

        void inverse_ntt_negacyclic_harvey_(
            sycl::queue& q,
            sycl::buffer<uint64_t> buf_operand,
            sycl::buffer<uint64_t>& buf_irp,
            sycl::buffer<uint64_t>& buf_sirp,
            uint64_t modulus, size_t n, bool lazy
        ){
          sycl::nd_range<1> work_items{sycl::range<1>
            (1024*((n/2+1023)/1024)), sycl::range<1>(1024)};

          q.submit([&](sycl::handler& cgh){
          auto _irp = buf_irp.get_access<sycl::access::mode::read>(cgh);
          auto _sirp = buf_sirp.get_access<sycl::access::mode::read>(cgh);
          auto _operand = buf_operand.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for<class _inverse_ntt_negacyclic_harvey>
          (work_items, [=](sycl::nd_item<1> it){
              int tid = it.get_group(0) * work_items.get_local_range().get(0) + it.get_local_id(0);
              if(2*tid < n){

              uint64_t two_times_modulus = modulus * 2;
              size_t t = 1;

              for (size_t m = n; m > 1; m >>= 1)
              {
                  size_t h = m >> 1;
                  size_t i = tid / t;
                  size_t local_id = tid % t;
                  size_t j1 = i * 2 * t;
                  const uint64_t W = _irp[h + i];
                  const uint64_t Wprime = _sirp[h + i];

                  uint64_t U = _operand[j1+local_id];
                  uint64_t V = _operand[j1+local_id+t];
                  uint64_t currU;
                  uint64_t T;
                  unsigned long long H;

                  T = two_times_modulus - V + U;
                  currU = U + V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((U << 1) >= T)));
                  U = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                  multiply_uint64_hw64(Wprime, T, &H);
                  V = T * W - H * modulus;
                  _operand[j1+local_id] = U;
                  _operand[j1+local_id+t] = V;

                  t <<= 1;
                  it.barrier();
              }
              if (!lazy){
                if (_operand[tid*2] >= modulus) _operand[tid*2] -= modulus;
                if (_operand[tid*2+1] >= modulus) _operand[tid*2+1] -= modulus;
              }
            }
          });
        });
      }

        void inverse_ntt_negacyclic_harvey_lazy__(uint64_t *operand,
            const uint64_t *inv_root_powers_div_two,
            const uint64_t *scaled_inv_root_powers_div_two,
            uint64_t modulus, size_t n)
        {
            uint64_t two_times_modulus = modulus * 2;
            size_t t = 1;

            for (size_t m = n; m > 1; m >>= 1)
            {
                size_t j1 = 0;
                size_t h = m >> 1;
                    for (size_t tid = 0; 2*tid < n; tid++)
                    {
                        size_t i = tid / t;
                        size_t local_id = tid % t;
                        size_t j1 = i * 2 * t;
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
                        multiply_uint64_hw64(Wprime, T, &H);
                        V = T * W - H * modulus;
                        operand[j1+local_id] = U;
                        operand[j1+local_id+t] = V;
                    }
                t <<= 1;
            }
        }

        // Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and Michael Naehrig).
        void inverse_ntt_negacyclic_harvey_lazy___(uint64_t *operand,
            const uint64_t *inv_root_powers_div_two, const uint64_t *scaled_inv_root_powers_div_two,
            uint64_t modulus, size_t n)
        {
            uint64_t two_times_modulus = modulus * 2;
            size_t t = 1;

            for (size_t m = n; m > 1; m >>= 1)
            {
                size_t j1 = 0;
                size_t h = m >> 1;
                if (t >= 4)
                {
                    for (size_t i = 0; i < h; i++)
                    {
                        size_t j2 = j1 + t;
                        // Need the powers of phi^{-1} in bit-reversed order
                        const uint64_t W = inv_root_powers_div_two[h + i];
                        const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

                        uint64_t *U = operand + j1;
                        uint64_t *V = U + t;
                        uint64_t currU;
                        uint64_t T;
                        unsigned long long H;
                        for (size_t j = j1; j < j2; j += 4)
                        {
                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;
                        }
                        j1 += (t << 1);
                    }
                }
                else
                {
                    for (size_t i = 0; i < h; i++)
                    {
                        size_t j2 = j1 + t;
                        // Need the powers of  phi^{-1} in bit-reversed order
                        const uint64_t W = inv_root_powers_div_two[h + i];
                        const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

                        uint64_t *U = operand + j1;
                        uint64_t *V = U + t;
                        uint64_t currU;
                        uint64_t T;
                        unsigned long long H;
                        for (size_t j = j1; j < j2; j++)
                        {
                            // U = x[i], V = x[i+m]

                            // Compute U - V + 2q
                            T = two_times_modulus - *V + *U;

                            // Cleverly check whether currU + currV >= two_times_modulus
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));

                            // Need to make it so that div2_uint_mod takes values that are > q.
                            //div2_uint_mod(U, modulusptr, coeff_uint64_count, U);
                            // We use also the fact that parity of currU is same as parity of T.
                            // Since our modulus is always so small that currU + masked_modulus < 2^64,
                            // we never need to worry about wrapping around when adding masked_modulus.
                            //uint64_t masked_modulus = modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                            //uint64_t carry = add_uint64(currU, masked_modulus, 0, &currU);
                            //currU += modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;

                            multiply_uint64_hw64(Wprime, T, &H);
                            // effectively, the next two multiply perform multiply modulo beta = 2**wordsize.
                            *V++ = W * T - H * modulus;
                        }
                        j1 += (t << 1);
                    }
                }
                t <<= 1;
            }
        }
    }
}
