#include "seal/util/smallnttcuda.cuh"
void switch_key_inplace(){
        // Temporary results
        Pointer<uint64_t> temp_poly[2] {
            allocate_zero_poly(2 * coeff_count, rns_mod_count, pool),
            allocate_zero_poly(2 * coeff_count, rns_mod_count, pool)
        };

        // RNS decomposition index = key index
        for (size_t i = 0; i < decomp_mod_count; i++)
        {
            // For each RNS decomposition, multiply with key data and sum up.
            auto local_small_poly_0(allocate_uint(coeff_count, pool));
            auto local_small_poly_1(allocate_uint(coeff_count, pool));
            auto local_small_poly_2(allocate_uint(coeff_count, pool));

            const uint64_t *local_encrypted_ptr = nullptr;
            set_uint_uint(
                target + i * coeff_count,
                coeff_count,
                local_small_poly_0.get());
            if (scheme == scheme_type::CKKS)
            {
                inverse_ntt_negacyclic_harvey(
                    local_small_poly_0.get(),
                    small_ntt_tables[i]);
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
                            local_small_poly_0.get(),
                            coeff_count,
                            local_small_poly_1.get());
                    }
                    else
                    {
                        modulo_poly_coeffs_63(
                            local_small_poly_0.get(),
                            coeff_count,
                            key_modulus[index],
                            local_small_poly_1.get());
                    }

                    // Lazy reduction, output in [0, 4q).
                    ntt_negacyclic_harvey_lazy(
                        local_small_poly_1.get(),
                        small_ntt_tables[index]);
                    local_encrypted_ptr = local_small_poly_1.get();
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
                            temp_poly[k].get()[(j * coeff_count + l) * 2],
                            local_wide_product[0],
                            &local_low_word);
                        temp_poly[k].get()[(j * coeff_count + l) * 2] =
                            local_low_word;
                        temp_poly[k].get()[(j * coeff_count + l) * 2 + 1] +=
                            local_wide_product[1] + local_carry;
                    }
                }
            }
        }

        // Results are now stored in temp_poly[k]
        // Modulus switching should be performed
        auto local_small_poly(allocate_uint(coeff_count, pool));
        for (size_t k = 0; k < 2; k++)
        {
            // Reduce (ct mod 4qk) mod qk
            uint64_t *temp_poly_ptr = temp_poly[k].get() +
                decomp_mod_count * coeff_count * 2;
            for (size_t l = 0; l < coeff_count; l++)
            {
                temp_poly_ptr[l] = barrett_reduce_128(
                    temp_poly_ptr + l * 2,
                    key_modulus[key_mod_count - 1]);
            }
            // Lazy reduction, they are then reduced mod qi
            uint64_t *temp_last_poly_ptr = temp_poly[k].get() + decomp_mod_count * coeff_count * 2;
            inverse_ntt_negacyclic_harvey_lazy(
                temp_last_poly_ptr,
                small_ntt_tables[key_mod_count - 1]);

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
                temp_poly_ptr = temp_poly[k].get() + j * coeff_count * 2;
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
                    local_small_poly.get());

                uint64_t half_mod = barrett_reduce_63(half, key_modulus[j]);
                for (size_t l = 0; l < coeff_count; l++)
                {
                    local_small_poly.get()[l] = sub_uint_uint_mod(local_small_poly.get()[l],
                        half_mod,
                        key_modulus[j]);
                }

                if (scheme == scheme_type::CKKS)
                {
                    cuda_ntt_negacyclic_harvey_(
                        local_small_poly.get(),
                        small_ntt_tables[j]);
                }
                else if (scheme == scheme_type::BFV)
                {
                    cuda_inverse_ntt_negacyclic_harvey_(
                        temp_poly_ptr,
                        small_ntt_tables[j]);
                }
                // ((ct mod qi) - (ct mod qk)) mod qi
                sub_poly_poly_coeffmod(
                    temp_poly_ptr,
                    local_small_poly.get(),
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
