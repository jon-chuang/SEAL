#include "seal/util/smallnttcuda.cuh"

using namespace seal;

void switch_key_inplace_(size_t& coeff_count, size_t& rns_mod_count,
    const uint64_t *target, seal::scheme_type& scheme,
    const seal::util::Pointer<seal::util::SmallNTTTables>& small_ntt_tables,
    size_t& key_mod_count, size_t& decomp_mod_count,
    const std::vector<seal::SmallModulus>& key_modulus,
    const std::vector<seal::PublicKey>& key_vector,
    seal::Ciphertext& encrypted,
    const seal::util::Pointer<long unsigned int, void>& modswitch_factors,
    seal::MemoryPoolHandle& pool)
{
        // Temporary results
        uint64_t* temp_poly[2][2 * coeff_count * rns_mod_count] {0};
        cudaMallocManaged(&temp_poly[0], 2 * coeff_count * rns_mod_count * sizeof(uint64_t));
        cudaMallocManaged(&temp_poly[1], 2 * coeff_count * rns_mod_count * sizeof(uint64_t));

        // RNS decomposition index = key index
        for (size_t i = 0; i < decomp_mod_count; i++)
        {
            // For each RNS decomposition, multiply with key data and sum up.
            uint64_t *local_small_poly_0, *local_small_poly_1, *local_small_poly_2,
            cudaMallocManaged(local_small_poly_0, coeff_count*sizeof(uint64_t));
            cudaMallocManaged(local_small_poly_1, coeff_count*sizeof(uint64_t));
            cudaMallocManaged(local_small_poly_2, coeff_count*sizeof(uint64_t));

            const uint64_t *local_encrypted_ptr = nullptr;
            set_uint_uint(
                target + i * coeff_count,
                coeff_count,
                local_small_poly_0;
            if (scheme == scheme_type::CKKS)
            {
                size_t n = size_t(1) << small_ntt_tables[i].coeff_count_power();
                const uint64_t *inv_root_powers_div_two =
                    small_ntt_tables[i].get_inv_root_powers_div_two();
                const uint64_t *scaled_inv_root_powers_div_two =
                    small_ntt_tables[i].get_scaled_inv_root_powers_div_two();
                uint64_t modulus = small_ntt_tables[i].modulus().value();

                size_t blocksize = min(n/2, size_t(1024));

                dim3 block_dim(blocksize, 1, 1);
                dim3 grid_dim(n/(2*blocksize), 1, 1);

                cuda_inverse_ntt_negacyclic_harvey<<<grid_dim, block_dim>>>(local_small_poly_0, inv_root_powers_div_two,
                  scaled_inv_root_powers_div_two, modulus, n);
            }
            // Key RNS representation
            for (size_t j = 0; j < rns_mod_count; j++)
            {
                size_t index = (j == decomp_mod_count ? key_mod_count - 1 : j);
                if (scheme == scheme_type::CKKS && i == j)
                {
                    local_encrypted_ptr = target + j * coeff_count;
                }
                else
                {
                    // Reduce modulus only if needed
                    if (key_modulus[i].value() <= key_modulus[index].value())
                    {
                        set_uint_uint(
                            local_small_poly_0,
                            coeff_count,
                            local_small_poly_1);
                    }
                    else
                    {
                        modulo_poly_coeffs_63(
                            local_small_poly_0,
                            coeff_count,
                            key_modulus[index],
                            local_small_poly_1);
                    }

                    // Lazy reduction, output in [0, 4q).

                    size_t n = size_t(1) << small_ntt_tables[index].coeff_count_power();
                    const uint64_t *root_powers = small_ntt_tables[index].get_root_powers();
                    const uint64_t *scaled_root_powers = small_ntt_tables[index].get_scaled_root_powers();
                    uint64_t modulus = small_ntt_tables[index].modulus().value();

                    size_t blocksize = min(n/2, size_t(1024));

                    dim3 block_dim(blocksize, 1, 1);
                    dim3 grid_dim(n/(2*blocksize), 1, 1);

                    cuda_ntt_negacyclic_harvey_lazy_<<<grid_dim, block_dim>>>(operand, root_powers,
                      scaled_root_powers, modulus, n);
                    local_encrypted_ptr = local_small_poly_1;
                }
                // Two components in key
                for (size_t k = 0; k < 2; k++)
                {
                    const uint64_t *key_ptr = key_vector[i].data().data(k);
                    for (size_t l = 0; l < coeff_count; l++)
                    {
                        unsigned long long local_wide_product[2];
                        unsigned long long local_low_word;
                        unsigned char local_carry;

                        multiply_uint64(
                            local_encrypted_ptr[l],
                            key_ptr[(index * coeff_count) + l],
                            local_wide_product);
                        local_carry = add_uint64(
                            temp_poly[k][(j * coeff_count + l) * 2],
                            local_wide_product[0],
                            &local_low_word);
                        temp_poly[k][(j * coeff_count + l) * 2] =
                            local_low_word;
                        temp_poly[k][(j * coeff_count + l) * 2 + 1] +=
                            local_wide_product[1] + local_carry;
                    }
                }
            }
        }

        // Results are now stored in temp_poly[k]
        // Modulus switching should be performed
        uint64_t *local_small_poly;
        cudaMallocManaged(&local_small_poly, coeff_count*sizeof(uint64_t));
        for (size_t k = 0; k < 2; k++)
        {
            // Reduce (ct mod 4qk) mod qk
            uint64_t *temp_poly_ptr = temp_poly[k] +
                decomp_mod_count * coeff_count * 2;
            for (size_t l = 0; l < coeff_count; l++)
            {
                temp_poly_ptr[l] = barrett_reduce_128(
                    temp_poly_ptr + l * 2,
                    key_modulus[key_mod_count - 1]);
            }
            // Lazy reduction, they are then reduced mod qi
            uint64_t *temp_last_poly_ptr = temp_poly[k] + decomp_mod_count * coeff_count * 2;

/// KERNEL CALL
            size_t n = size_t(1) << small_ntt_tables[key_mod_count - 1].coeff_count_power();
            const uint64_t *inv_root_powers_div_two =
                small_ntt_tables[key_mod_count - 1].get_inv_root_powers_div_two();
            const uint64_t *scaled_inv_root_powers_div_two =
                small_ntt_tables[key_mod_count - 1].get_scaled_inv_root_powers_div_two();
            uint64_t modulus = small_ntt_tables[key_mod_count - 1].modulus().value();

            size_t blocksize = min(n/2, size_t(1024));

            dim3 block_dim(blocksize, 1, 1);
            dim3 grid_dim(n/(2*blocksize), 1, 1);

            cuda_inverse_ntt_negacyclic_harvey_lazy<<<grid_dim, block_dim>>>
                (temp_last_poly_ptr, inv_root_powers_div_two,
                scaled_inv_root_powers_div_two, modulus, n);
/// KERNEL CALL END

            // Add (p-1)/2 to change from flooring to rounding.
            uint64_t half = key_modulus[key_mod_count - 1].value() >> 1;
            for (size_t l = 0; l < coeff_count; l++)
            {
                temp_last_poly_ptr[l] = barrett_reduce_63(temp_last_poly_ptr[l] + half,
                    key_modulus[key_mod_count - 1]);
            }

            uint64_t *encrypted_ptr = encrypted.data(k);
            for (size_t j = 0; j < decomp_mod_count; j++)
            {
                temp_poly_ptr = temp_poly[k] + j * coeff_count * 2;
                // (ct mod 4qi) mod qi
                for (size_t l = 0; l < coeff_count; l++)
                {
                    temp_poly_ptr[l] = barrett_reduce_128(
                        temp_poly_ptr + l * 2,
                        key_modulus[j]);
                }
                // (ct mod 4qk) mod qi
                modulo_poly_coeffs_63(
                    temp_last_poly_ptr,
                    coeff_count,
                    key_modulus[j],
                    local_small_poly);

                uint64_t half_mod = barrett_reduce_63(half, key_modulus[j]);
                for (size_t l = 0; l < coeff_count; l++)
                {
                    local_small_poly[l] = sub_uint_uint_mod(local_small_poly[l],
                        half_mod,
                        key_modulus[j]);
                }

                if (scheme == scheme_type::CKKS)
                {
                    size_t n = size_t(1) << small_ntt_tables[j].coeff_count_power();
                    const uint64_t *root_powers = small_ntt_tables[j].get_root_powers();
                    const uint64_t *scaled_root_powers = small_ntt_tables[j].get_scaled_root_powers();
                    uint64_t modulus = small_ntt_tables[j].modulus().value();
                    cuda_ntt_negacyclic_harvey_<<<grid_dim, block_dim>>>
                      (operand, root_powers, scaled_root_powers, modulus, n);
                }
                else if (scheme == scheme_type::BFV)
                {
                    size_t n = size_t(1) << small_ntt_tables[j].coeff_count_power();
                    const uint64_t *inv_root_powers_div_two =
                        small_ntt_tables[j].get_inv_root_powers_div_two();
                    const uint64_t *scaled_inv_root_powers_div_two =
                      small_ntt_tables[j].get_scaled_inv_root_powers_div_two();
                    uint64_t modulus = small_ntt_tables[j].modulus().value();

                    size_t blocksize = min(n/2, size_t(1024));

                    dim3 block_dim(blocksize, 1, 1);
                    dim3 grid_dim(n/(2*blocksize), 1, 1);

                    cuda_inverse_ntt_negacyclic_harvey<<<grid_dim, block_dim>>>(temp_poly_ptr, inv_root_powers_div_two,
                      scaled_inv_root_powers_div_two, modulus, n);
                }
                // ((ct mod qi) - (ct mod qk)) mod qi
                sub_poly_poly_coeffmod(
                    temp_poly_ptr,
                    local_small_poly,
                    coeff_count,
                    key_modulus[j],
                    temp_poly_ptr);
                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(
                    temp_poly_ptr,
                    coeff_count,
                    modswitch_factors[j],
                    key_modulus[j],
                    temp_poly_ptr);
                add_poly_poly_coeffmod(
                    temp_poly_ptr,
                    encrypted_ptr + j * coeff_count,
                    coeff_count,
                    key_modulus[j],
                    encrypted_ptr + j * coeff_count);
            }
        }
}
