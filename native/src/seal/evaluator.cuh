#include <cuda_runtime.h>
#include "seal/smallmodulus.h"

namespace seal
{
  namespace util
  {
    void switch_key_inplace_(size_t& coeff_count, size_t& rns_mod_count,
        const uint64_t *target, seal::scheme_type& scheme,
        const seal::util::Pointer<seal::util::SmallNTTTables>& small_ntt_tables,
        size_t& key_mod_count, size_t& decomp_mod_count,
        const std::vector<seal::SmallModulus>& key_modulus,
        const std::vector<seal::PublicKey>& key_vector,
        seal::Ciphertext& encrypted,
        const seal::util::Pointer<long unsigned int, void>& modswitch_factors,
        seal::MemoryPoolHandle& pool);

        template<typename T, typename S, typename = std::enable_if<is_uint64_v<T, S>>>
        void multiply_uint64_(T operand1, S operand2,
            unsigned long long *result128)
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

            result128[1] = static_cast<unsigned long long>(
                left + (middle >> 32) + (temp_sum >> 32));
            result128[0] = static_cast<unsigned long long>(
                (temp_sum << 32) | (right & 0x00000000FFFFFFFF));
        }

        template<typename T>
        std::uint64_t barrett_reduce_63_(
                    T input, const std::uint64_t modulus,
                    const std::uint64_t *const_ratio)
        {
            // Reduces input using base 2^64 Barrett reduction
            // input must be at most 63 bits

            unsigned long long tmp[2];
            multiply_uint64_(input, const_ratio[1], tmp);

            // Barrett subtraction
            tmp[0] = input - tmp[1] * modulus;

            // One more subtraction is enough
            return static_cast<std::uint64_t>(tmp[0]) -
                (modulus & static_cast<std::uint64_t>(
                    -static_cast<std::int64_t>(tmp[0] >= modulus)));
        }

        std::uint64_t modulo_poly_coeffs_63_(const std::uint64_t *poly,
                    std::size_t coeff_count, const std::uint64_t modulus,
                    const std::uint64_t *const_ratio,
                    std::uint64_t *result)
        {
            // This function is the fastest for reducing polynomial coefficients,
            // but requires that the input coefficients are at most 63 bits, unlike
            // modulo_poly_coeffs that allows also 64-bit coefficients.
            std::transform(poly, poly + coeff_count, result,
                [&](auto coeff) {
                    return barrett_reduce_63_(coeff, modulus, const_ratio);
                });
        }
    }
}
