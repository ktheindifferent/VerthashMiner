/*
 * Copyright 2021 CryptoGraphics
 * Metal Shading Language port for Apple GPUs
 */

#include <metal_stdlib>
using namespace metal;

// Metal rotate function (64-bit)
inline uint64_t rotr64(uint64_t x, uint32_t n) {
    return rotate(x, (uint64_t)(64-n));
}

inline uint32_t rotl32(uint32_t x, uint32_t n) {
    return (((x) << (n)) | ((x) >> (32 - (n))));
}

inline uint32_t fnv1a(const uint32_t a, const uint32_t b) {
    uint32_t res = (a ^ b) * 0x1000193U;
    return res;
}

// Keccak constants
constant uint64_t keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

//-----------------------------------------------------------------------------
// SHA3-512 Precompute kernel
// Runs once per work unit to compute 8 keccak states from block header
//-----------------------------------------------------------------------------
kernel void sha3_512_precompute(
    device uint64_t* output [[buffer(0)]],
    device uint32_t* header [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
)
{
    uint32_t h0 = header[0] + (gid & 7) + 1;

    // sha3 init
    uint64_t st[25] = { 0 };

    // sha3 update (header 72 bytes)
    st[0] = as_type<uint64_t>(uint2(h0,         header[ 1]));
    st[1] = as_type<uint64_t>(uint2(header[ 2], header[ 3]));
    st[2] = as_type<uint64_t>(uint2(header[ 4], header[ 5]));
    st[3] = as_type<uint64_t>(uint2(header[ 6], header[ 7]));
    st[4] = as_type<uint64_t>(uint2(header[ 8], header[ 9]));
    st[5] = as_type<uint64_t>(uint2(header[10], header[11]));
    st[6] = as_type<uint64_t>(uint2(header[12], header[13]));
    st[7] = as_type<uint64_t>(uint2(header[14], header[15]));
    st[8] = as_type<uint64_t>(uint2(header[16], header[17]));

    uint64_t u[5];
    uint64_t v, w;

    for (int r = 0; r < 24; ++r)
    {
        // Theta
        v    = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        u[2] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        u[3] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        u[4] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        w    = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        u[0] = rotr64(u[2], 63) ^    w;
        u[1] = rotr64(u[3], 63) ^    v;
        u[2] = rotr64(u[4], 63) ^ u[2];
        u[3] = rotr64(   w, 63) ^ u[3];
        u[4] = rotr64(   v, 63) ^ u[4];

        st[0] ^= u[0]; st[5] ^= u[0]; st[10] ^= u[0]; st[15] ^= u[0]; st[20] ^= u[0];
        st[1] ^= u[1]; st[6] ^= u[1]; st[11] ^= u[1]; st[16] ^= u[1]; st[21] ^= u[1];
        st[2] ^= u[2]; st[7] ^= u[2]; st[12] ^= u[2]; st[17] ^= u[2]; st[22] ^= u[2];
        st[3] ^= u[3]; st[8] ^= u[3]; st[13] ^= u[3]; st[18] ^= u[3]; st[23] ^= u[3];
        st[4] ^= u[4]; st[9] ^= u[4]; st[14] ^= u[4]; st[19] ^= u[4]; st[24] ^= u[4];

        // Rho Pi
        v = st[1];
        st[ 1] = rotr64(st[ 6], 20);
        st[ 6] = rotr64(st[ 9], 44);
        st[ 9] = rotr64(st[22],  3);
        st[22] = rotr64(st[14], 25);
        st[14] = rotr64(st[20], 46);
        st[20] = rotr64(st[ 2],  2);
        st[ 2] = rotr64(st[12], 21);
        st[12] = rotr64(st[13], 39);
        st[13] = rotr64(st[19], 56);
        st[19] = rotr64(st[23],  8);
        st[23] = rotr64(st[15], 23);
        st[15] = rotr64(st[ 4], 37);
        st[ 4] = rotr64(st[24], 50);
        st[24] = rotr64(st[21], 62);
        st[21] = rotr64(st[ 8],  9);
        st[ 8] = rotr64(st[16], 19);
        st[16] = rotr64(st[ 5], 28);
        st[ 5] = rotr64(st[ 3], 36);
        st[ 3] = rotr64(st[18], 43);
        st[18] = rotr64(st[17], 49);
        st[17] = rotr64(st[11], 54);
        st[11] = rotr64(st[ 7], 58);
        st[ 7] = rotr64(st[10], 61);
        st[10] = rotr64(v, 63);

        // Chi (Metal uses select instead of bitselect, with inverted condition)
        v = st[ 0]; w = st[ 1]; st[ 0] = select(st[ 0] ^ st[ 2], st[ 0], ~st[ 1]); st[ 1] = select(st[ 1] ^ st[ 3], st[ 1], ~st[ 2]); st[ 2] = select(st[ 2] ^ st[ 4], st[ 2], ~st[ 3]); st[ 3] = select(st[ 3] ^ v, st[ 3], ~st[ 4]); st[ 4] = select(st[ 4] ^ w, st[ 4], ~v);
        v = st[ 5]; w = st[ 6]; st[ 5] = select(st[ 5] ^ st[ 7], st[ 5], ~st[ 6]); st[ 6] = select(st[ 6] ^ st[ 8], st[ 6], ~st[ 7]); st[ 7] = select(st[ 7] ^ st[ 9], st[ 7], ~st[ 8]); st[ 8] = select(st[ 8] ^ v, st[ 8], ~st[ 9]); st[ 9] = select(st[ 9] ^ w, st[ 9], ~v);
        v = st[10]; w = st[11]; st[10] = select(st[10] ^ st[12], st[10], ~st[11]); st[11] = select(st[11] ^ st[13], st[11], ~st[12]); st[12] = select(st[12] ^ st[14], st[12], ~st[13]); st[13] = select(st[13] ^ v, st[13], ~st[14]); st[14] = select(st[14] ^ w, st[14], ~v);
        v = st[15]; w = st[16]; st[15] = select(st[15] ^ st[17], st[15], ~st[16]); st[16] = select(st[16] ^ st[18], st[16], ~st[17]); st[17] = select(st[17] ^ st[19], st[17], ~st[18]); st[18] = select(st[18] ^ v, st[18], ~st[19]); st[19] = select(st[19] ^ w, st[19], ~v);
        v = st[20]; w = st[21]; st[20] = select(st[20] ^ st[22], st[20], ~st[21]); st[21] = select(st[21] ^ st[23], st[21], ~st[22]); st[22] = select(st[22] ^ st[24], st[22], ~st[23]); st[23] = select(st[23] ^ v, st[23], ~st[24]); st[24] = select(st[24] ^ w, st[24], ~v);

        // Iota
        st[0] ^= keccakf_rndc[r];
    }

    // store result - each thread stores 25 uint64s
    device uint64_t* kstate = output + (25 * (gid & 7));
    for (int i = 0; i < 25; ++i) {
        kstate[i] = st[i];
    }
}

//-----------------------------------------------------------------------------
// SHA3-512-256 kernel
// Computes SHA3-256 hash of header for each nonce
//-----------------------------------------------------------------------------
kernel void sha3_512_256(
    device uint32_t* output [[buffer(0)]],
    device uint32_t* header [[buffer(1)]],
    constant uint32_t& in18 [[buffer(2)]],
    constant uint32_t& firstNonce [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    uint32_t nonce = firstNonce + gid;

    // sha3 init
    uint64_t st[25] = { 0 };

    // sha3 update (header 72 bytes)
    st[0] ^= as_type<uint64_t>(uint2(header[ 0], header[ 1]));
    st[1] ^= as_type<uint64_t>(uint2(header[ 2], header[ 3]));
    st[2] ^= as_type<uint64_t>(uint2(header[ 4], header[ 5]));
    st[3] ^= as_type<uint64_t>(uint2(header[ 6], header[ 7]));
    st[4] ^= as_type<uint64_t>(uint2(header[ 8], header[ 9]));
    st[5] ^= as_type<uint64_t>(uint2(header[10], header[11]));
    st[6] ^= as_type<uint64_t>(uint2(header[12], header[13]));
    st[7] ^= as_type<uint64_t>(uint2(header[14], header[15]));
    st[8] ^= as_type<uint64_t>(uint2(header[16], header[17]));

    // output size 32 bytes
    st[9] ^= as_type<uint64_t>(uint2(in18, nonce));
    st[10] ^= 0x06UL;
    st[16] ^= 0x8000000000000000UL;

    uint64_t u[5];
    uint64_t v, w;

    for (int r = 0; r < 24; r++)
    {
        // Theta
        v    = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        u[2] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        u[3] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        u[4] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        w    = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        u[0] = rotr64(u[2], 63) ^    w;
        u[1] = rotr64(u[3], 63) ^    v;
        u[2] = rotr64(u[4], 63) ^ u[2];
        u[3] = rotr64(   w, 63) ^ u[3];
        u[4] = rotr64(   v, 63) ^ u[4];

        st[0] ^= u[0]; st[5] ^= u[0]; st[10] ^= u[0]; st[15] ^= u[0]; st[20] ^= u[0];
        st[1] ^= u[1]; st[6] ^= u[1]; st[11] ^= u[1]; st[16] ^= u[1]; st[21] ^= u[1];
        st[2] ^= u[2]; st[7] ^= u[2]; st[12] ^= u[2]; st[17] ^= u[2]; st[22] ^= u[2];
        st[3] ^= u[3]; st[8] ^= u[3]; st[13] ^= u[3]; st[18] ^= u[3]; st[23] ^= u[3];
        st[4] ^= u[4]; st[9] ^= u[4]; st[14] ^= u[4]; st[19] ^= u[4]; st[24] ^= u[4];

        // Rho Pi
        v = st[1];
        st[ 1] = rotr64(st[ 6], 20);
        st[ 6] = rotr64(st[ 9], 44);
        st[ 9] = rotr64(st[22],  3);
        st[22] = rotr64(st[14], 25);
        st[14] = rotr64(st[20], 46);
        st[20] = rotr64(st[ 2],  2);
        st[ 2] = rotr64(st[12], 21);
        st[12] = rotr64(st[13], 39);
        st[13] = rotr64(st[19], 56);
        st[19] = rotr64(st[23],  8);
        st[23] = rotr64(st[15], 23);
        st[15] = rotr64(st[ 4], 37);
        st[ 4] = rotr64(st[24], 50);
        st[24] = rotr64(st[21], 62);
        st[21] = rotr64(st[ 8],  9);
        st[ 8] = rotr64(st[16], 19);
        st[16] = rotr64(st[ 5], 28);
        st[ 5] = rotr64(st[ 3], 36);
        st[ 3] = rotr64(st[18], 43);
        st[18] = rotr64(st[17], 49);
        st[17] = rotr64(st[11], 54);
        st[11] = rotr64(st[ 7], 58);
        st[ 7] = rotr64(st[10], 61);
        st[10] = rotr64(v, 63);

        // Chi
        v = st[ 0]; w = st[ 1]; st[ 0] = select(st[ 0] ^ st[ 2], st[ 0], ~st[ 1]); st[ 1] = select(st[ 1] ^ st[ 3], st[ 1], ~st[ 2]); st[ 2] = select(st[ 2] ^ st[ 4], st[ 2], ~st[ 3]); st[ 3] = select(st[ 3] ^ v, st[ 3], ~st[ 4]); st[ 4] = select(st[ 4] ^ w, st[ 4], ~v);
        v = st[ 5]; w = st[ 6]; st[ 5] = select(st[ 5] ^ st[ 7], st[ 5], ~st[ 6]); st[ 6] = select(st[ 6] ^ st[ 8], st[ 6], ~st[ 7]); st[ 7] = select(st[ 7] ^ st[ 9], st[ 7], ~st[ 8]); st[ 8] = select(st[ 8] ^ v, st[ 8], ~st[ 9]); st[ 9] = select(st[ 9] ^ w, st[ 9], ~v);
        v = st[10]; w = st[11]; st[10] = select(st[10] ^ st[12], st[10], ~st[11]); st[11] = select(st[11] ^ st[13], st[11], ~st[12]); st[12] = select(st[12] ^ st[14], st[12], ~st[13]); st[13] = select(st[13] ^ v, st[13], ~st[14]); st[14] = select(st[14] ^ w, st[14], ~v);
        v = st[15]; w = st[16]; st[15] = select(st[15] ^ st[17], st[15], ~st[16]); st[16] = select(st[16] ^ st[18], st[16], ~st[17]); st[17] = select(st[17] ^ st[19], st[17], ~st[18]); st[18] = select(st[18] ^ v, st[18], ~st[19]); st[19] = select(st[19] ^ w, st[19], ~v);
        v = st[20]; w = st[21]; st[20] = select(st[20] ^ st[22], st[20], ~st[21]); st[21] = select(st[21] ^ st[23], st[21], ~st[22]); st[22] = select(st[22] ^ st[24], st[22], ~st[23]); st[23] = select(st[23] ^ v, st[23], ~st[24]); st[24] = select(st[24] ^ w, st[24], ~v);

        // Iota
        st[0] ^= keccakf_rndc[r];
    }

    // Store 256-bit (8 x uint32) result
    device uint32_t* hash = output + (8 * gid);
    hash[0] = as_type<uint2>(st[0]).x;
    hash[1] = as_type<uint2>(st[0]).y;
    hash[2] = as_type<uint2>(st[1]).x;
    hash[3] = as_type<uint2>(st[1]).y;
    hash[4] = as_type<uint2>(st[2]).x;
    hash[5] = as_type<uint2>(st[2]).y;
    hash[6] = as_type<uint2>(st[3]).x;
    hash[7] = as_type<uint2>(st[3]).y;
}

//-----------------------------------------------------------------------------
// 2x precomputed SHA3 states (for 4-way kernel)
//-----------------------------------------------------------------------------
struct kstate2x_t {
    uint64_t ul[50];
};

// shared hash to exchange between lanes during memory seeks stage
struct hash8_t {
    uint2 u2[4];
};

// A combined SHA3 result used during memory seeks stage
struct sha3_state_t {
    uint32_t u[128];
};

//-----------------------------------------------------------------------------
// Verthash 4-way kernel
// Main memory-hard hash computation
//-----------------------------------------------------------------------------
kernel void verthash_4w(
    device uint2* io_hashes [[buffer(0)]],
    device kstate2x_t* kStates [[buffer(1)]],
    device uint2* memory [[buffer(2)]],
    constant uint32_t& in18 [[buffer(3)]],
    constant uint32_t& firstNonce [[buffer(4)]],
    device atomic_uint* targetResults [[buffer(5)]],
    constant uint64_t& target [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
)
{
    // WORK_SIZE is configured at compile time (256)
    constexpr uint32_t WORK_SIZE = 256;
    constexpr uint32_t MDIV = 71303125;  // Verthash MDIV constant

    // 4x lane group index(local)
    uint32_t lgr4id = lid >> 2;
    // 4x lane group index(global) used as nonce result
    uint32_t gr4id = gid >> 2;
    // sub group id(of 4x lane group)
    uint32_t gr4e = gid & 3;

    //-----------------------------------------------------------------------------
    // SHA3 stage
    device kstate2x_t* kstate = &kStates[gr4e];

    threadgroup sha3_state_t sha3St[WORK_SIZE/4];
    uint32_t nonce = firstNonce + gr4id;

    // 4 way kernel running 8xSHA3 passes(2x each lane)
    for(int s3s = 0; s3s < 2; ++s3s)
    {
        uint64_t st[25] = { 0 };
        // load state
        for(int i = 0; i < 25; ++i)
        {
            st[i] = kstate->ul[25 * s3s + i];
        }

        // variables
        st[0] ^= as_type<uint64_t>(uint2(in18, nonce));

        st[1] ^= 0x06UL;
        st[8] ^= 0x8000000000000000UL;

        uint64_t u[5];
        uint64_t v,w;

        for (int r = 0; r < 24; r++)
        {
            // Theta
            v    = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
            u[2] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
            u[3] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
            u[4] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
            w    = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

            u[0] = rotr64(u[2], 63) ^    w;
            u[1] = rotr64(u[3], 63) ^    v;
            u[2] = rotr64(u[4], 63) ^ u[2];
            u[3] = rotr64(   w, 63) ^ u[3];
            u[4] = rotr64(   v, 63) ^ u[4];

            st[0] ^= u[0]; st[5] ^= u[0]; st[10] ^= u[0]; st[15] ^= u[0]; st[20] ^= u[0];
            st[1] ^= u[1]; st[6] ^= u[1]; st[11] ^= u[1]; st[16] ^= u[1]; st[21] ^= u[1];
            st[2] ^= u[2]; st[7] ^= u[2]; st[12] ^= u[2]; st[17] ^= u[2]; st[22] ^= u[2];
            st[3] ^= u[3]; st[8] ^= u[3]; st[13] ^= u[3]; st[18] ^= u[3]; st[23] ^= u[3];
            st[4] ^= u[4]; st[9] ^= u[4]; st[14] ^= u[4]; st[19] ^= u[4]; st[24] ^= u[4];

            // Rho Pi
            v = st[1];
            st[ 1] = rotr64(st[ 6], 20);
            st[ 6] = rotr64(st[ 9], 44);
            st[ 9] = rotr64(st[22],  3);
            st[22] = rotr64(st[14], 25);
            st[14] = rotr64(st[20], 46);
            st[20] = rotr64(st[ 2],  2);
            st[ 2] = rotr64(st[12], 21);
            st[12] = rotr64(st[13], 39);
            st[13] = rotr64(st[19], 56);
            st[19] = rotr64(st[23],  8);
            st[23] = rotr64(st[15], 23);
            st[15] = rotr64(st[ 4], 37);
            st[ 4] = rotr64(st[24], 50);
            st[24] = rotr64(st[21], 62);
            st[21] = rotr64(st[ 8],  9);
            st[ 8] = rotr64(st[16], 19);
            st[16] = rotr64(st[ 5], 28);
            st[ 5] = rotr64(st[ 3], 36);
            st[ 3] = rotr64(st[18], 43);
            st[18] = rotr64(st[17], 49);
            st[17] = rotr64(st[11], 54);
            st[11] = rotr64(st[ 7], 58);
            st[ 7] = rotr64(st[10], 61);
            st[10] = rotr64(v, 63);

            //  Chi (Metal uses select instead of bitselect, with inverted condition)
            v = st[ 0]; w = st[ 1]; st[ 0] = select(st[ 0] ^ st[ 2], st[ 0], ~st[ 1]); st[ 1] = select(st[ 1] ^ st[ 3], st[ 1], ~st[ 2]); st[ 2] = select(st[ 2] ^ st[ 4], st[ 2], ~st[ 3]); st[ 3] = select(st[ 3] ^ v, st[ 3], ~st[ 4]); st[ 4] = select(st[ 4] ^ w, st[ 4], ~v);
            v = st[ 5]; w = st[ 6]; st[ 5] = select(st[ 5] ^ st[ 7], st[ 5], ~st[ 6]); st[ 6] = select(st[ 6] ^ st[ 8], st[ 6], ~st[ 7]); st[ 7] = select(st[ 7] ^ st[ 9], st[ 7], ~st[ 8]); st[ 8] = select(st[ 8] ^ v, st[ 8], ~st[ 9]); st[ 9] = select(st[ 9] ^ w, st[ 9], ~v);
            v = st[10]; w = st[11]; st[10] = select(st[10] ^ st[12], st[10], ~st[11]); st[11] = select(st[11] ^ st[13], st[11], ~st[12]); st[12] = select(st[12] ^ st[14], st[12], ~st[13]); st[13] = select(st[13] ^ v, st[13], ~st[14]); st[14] = select(st[14] ^ w, st[14], ~v);
            v = st[15]; w = st[16]; st[15] = select(st[15] ^ st[17], st[15], ~st[16]); st[16] = select(st[16] ^ st[18], st[16], ~st[17]); st[17] = select(st[17] ^ st[19], st[17], ~st[18]); st[18] = select(st[18] ^ v, st[18], ~st[19]); st[19] = select(st[19] ^ w, st[19], ~v);
            v = st[20]; w = st[21]; st[20] = select(st[20] ^ st[22], st[20], ~st[21]); st[21] = select(st[21] ^ st[23], st[21], ~st[22]); st[22] = select(st[22] ^ st[24], st[22], ~st[23]); st[23] = select(st[23] ^ v, st[23], ~st[24]); st[24] = select(st[24] ^ w, st[24], ~v);

            //  Iota
            st[0] ^= keccakf_rndc[r];
        }

        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  0] = as_type<uint2>(st[0]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  1] = as_type<uint2>(st[0]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  2] = as_type<uint2>(st[1]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  3] = as_type<uint2>(st[1]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  4] = as_type<uint2>(st[2]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  5] = as_type<uint2>(st[2]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  6] = as_type<uint2>(st[3]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  7] = as_type<uint2>(st[3]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  8] = as_type<uint2>(st[4]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) +  9] = as_type<uint2>(st[4]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 10] = as_type<uint2>(st[5]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 11] = as_type<uint2>(st[5]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 12] = as_type<uint2>(st[6]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 13] = as_type<uint2>(st[6]).y;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 14] = as_type<uint2>(st[7]).x;
        sha3St[lgr4id].u[(gr4e * 32) + (s3s * 16) + 15] = as_type<uint2>(st[7]).y;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);


    //-----------------------------------------------------------------------------
    // Verthash IO memory seek stage

    // get SHA3 256 input
    uint2 up1;
    up1 = io_hashes[gid];

    // local array used to sync between lanes
    threadgroup hash8_t sHash[WORK_SIZE/4];

    uint32_t value_accumulator = 0x811c9dc5;

    for(uint32_t i = 0; i < 4096; ++i)
    {
        // v1. Rotate by constant amount
        uint32_t s3idx0 = i & 127;
        uint32_t seek_index = sha3St[lgr4id].u[s3idx0];
        uint32_t state0mod = rotl32(seek_index, 1);
        sha3St[lgr4id].u[s3idx0] = state0mod;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint32_t offset = (fnv1a(seek_index, value_accumulator) % MDIV) << 1;

        // 4 way memory lookup
        const uint2 vvalue = memory[offset + gr4e];

        // update up1
        up1.x = fnv1a(up1.x, vvalue.x);
        up1.y = fnv1a(up1.y, vvalue.y);

        // update value accumulator and synchronize it between lanes
        sHash[lgr4id].u2[gr4e] = vvalue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint2 uu0 = sHash[lgr4id].u2[0];
        uint2 uu1 = sHash[lgr4id].u2[1];
        uint2 uu2 = sHash[lgr4id].u2[2];
        uint2 uu3 = sHash[lgr4id].u2[3];
        value_accumulator = fnv1a(value_accumulator, uu0.x);
        value_accumulator = fnv1a(value_accumulator, uu0.y);
        value_accumulator = fnv1a(value_accumulator, uu1.x);
        value_accumulator = fnv1a(value_accumulator, uu1.y);
        value_accumulator = fnv1a(value_accumulator, uu2.x);
        value_accumulator = fnv1a(value_accumulator, uu2.y);
        value_accumulator = fnv1a(value_accumulator, uu3.x);
        value_accumulator = fnv1a(value_accumulator, uu3.y);
    }

    // store result
    io_hashes[gid] = up1;

    //---------------------------------------------------
    // Save result as HTarg (using extended validation)
    if(gr4e == 3)
    {
        uint64_t up1_64 = as_type<uint64_t>(up1);
        if(up1_64 <= target)
        {
            uint32_t ai = atomic_fetch_add_explicit(targetResults, 1, memory_order_relaxed);
            atomic_store_explicit(&targetResults[ai+1], gr4id, memory_order_relaxed); // final nonce
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
}
