// A library of arithmetic functions which have been stripped of the unnecessary OOP fat.
#pragma once

#include "seal/util/uintarith.h"

namespace seal
{
    namespace util
    {
        template<typename T>
                SEAL_NODISCARD inline std::uint64_t barrett_reduce_63_(
                    T input, const std::uint64_t modulus, const std::uint64_t const_ratio)
                {
                    unsigned long long tmp[2];
                    multiply_uint64(input, const_ratio, tmp);
                    // Barrett subtraction
                    tmp[0] = input - tmp[1] * modulus;

                    // One more subtraction is enough
                    return static_cast<std::uint64_t>(tmp[0]) -
                        (modulus & static_cast<std::uint64_t>(
                            -static_cast<std::int64_t>(tmp[0] >= modulus)));
                }
    }
}
