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
    }
}
