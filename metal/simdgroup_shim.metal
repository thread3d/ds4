// Portable simdgroup_matrix fallback for GPUs without hardware matrix support.
//
// Metal only lowers the `simdgroup_matrix` builtins on Apple GPUs (the
// MTLGPUFamilyApple7 feature set).  On other Metal 3 devices - notably the
// AMD Radeon cards in Intel Macs - the shader compiler accepts the MSL source
// and then fails to build any pipeline that uses them:
//
//     SC compilation failure
//     There is a call to an undefined label
//
// DwarfStar's kernels use a very small slice of the API (8x8 load/store,
// multiply, multiply-accumulate, make_filled).  This file reimplements exactly
// that slice on top of portable SIMD-group operations, which AMD Metal 3 does
// support.  It is enabled only when the device is not Apple7 hardware, so the
// hardware path stays in place on the machines this engine targets.
//
// Layout: an 8x8 matrix is spread over the 32 lanes of a SIMD group as two
// elements per lane.  Lane l holds (row r, columns 2j and 2j+1) where
// r = l / 4 and j = l % 4.  The mapping is internal to the shim: both the
// loads/stores and the multiply-accumulate use it, and the memory addressing
// matches Apple's documented `elements_per_row` / `matrix_origin` /
// `transpose_matrix` semantics so the surrounding kernels are unchanged.

#ifdef DS4_METAL_SIMDGROUP_SHIM

namespace ds4_sg {

inline ushort lane_id() {
    // MSL does not expose the SIMD lane id to non-entry-point functions, but
    // an exclusive prefix sum of 1 is exactly the lane id.
    return (ushort)simd_prefix_exclusive_sum((ushort)1);
}

template <typename T, int Cols = 8, int Rows = Cols>
struct mat {
    vec<T, 2> e;

    mat() thread {}

    // The hardware type has a diagonal-fill constructor for square matrices.
    explicit mat(T value) thread {
        const ushort l = lane_id();
        const uint r = l >> 2;
        const uint j = l & 3u;
        e[0] = (2u * j == r) ? value : (T)0;
        e[1] = (2u * j + 1u == r) ? value : (T)0;
    }
};

// Element (r, c) of the tile lives at src[(y + row) * elements_per_row + x + col]
// for a normal load, and at src[(y + c) * elements_per_row + x + r] when the
// tile is transposed.
template <typename T, typename Ptr>
inline void load(thread mat<T> &d, Ptr src, ulong elements_per_row = 8,
                 ulong2 matrix_origin = ulong2(0, 0), bool transpose_matrix = false) {
    const ushort l = lane_id();
    const uint r = l >> 2;
    const uint j = l & 3u;
    const ulong x = matrix_origin.x;
    const ulong y = matrix_origin.y;
    if (!transpose_matrix) {
        d.e[0] = src[(y + r) * elements_per_row + x + 2u * j];
        d.e[1] = src[(y + r) * elements_per_row + x + 2u * j + 1u];
    } else {
        d.e[0] = src[(y + 2u * j) * elements_per_row + x + r];
        d.e[1] = src[(y + 2u * j + 1u) * elements_per_row + x + r];
    }
}

template <typename T, typename Ptr>
inline void store(mat<T> a, Ptr dst, ulong elements_per_row = 8,
                  ulong2 matrix_origin = ulong2(0, 0), bool transpose_matrix = false) {
    const ushort l = lane_id();
    const uint r = l >> 2;
    const uint j = l & 3u;
    const ulong x = matrix_origin.x;
    const ulong y = matrix_origin.y;
    if (!transpose_matrix) {
        dst[(y + r) * elements_per_row + x + 2u * j] = a.e[0];
        dst[(y + r) * elements_per_row + x + 2u * j + 1u] = a.e[1];
    } else {
        dst[(y + 2u * j) * elements_per_row + x + r] = a.e[0];
        dst[(y + 2u * j + 1u) * elements_per_row + x + r] = a.e[1];
    }
}

// d = a * b + c.  The multiply-accumulate is the only cross-lane operation:
// every lane gathers the A column and B columns it needs with simd_shuffle.
// Gathers are performed in float so half and float operands share one path.
template <typename R, typename T, typename U, typename V>
inline void mac(thread mat<R> &d, mat<T> a, mat<U> b, mat<V> c) {
    const ushort l = lane_id();
    const uint r = l >> 2;
    const uint j = l & 3u;
    float acc0 = (float)c.e[0];
    float acc1 = (float)c.e[1];
    float a0 = (float)a.e[0];
    float a1 = (float)a.e[1];
    float b0 = (float)b.e[0];
    float b1 = (float)b.e[1];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const ushort la = (ushort)(r * 4u + (uint)(k >> 1));
        const float av = (k & 1) ? simd_shuffle(a1, la) : simd_shuffle(a0, la);
        const ushort lb = (ushort)((uint)k * 4u + j);
        acc0 += av * simd_shuffle(b0, lb);
        acc1 += av * simd_shuffle(b1, lb);
    }
    d.e[0] = (R)acc0;
    d.e[1] = (R)acc1;
}

template <typename R, typename T, typename U>
inline void mul(thread mat<R> &d, mat<T> a, mat<U> b) {
    mat<R> zero;
    zero.e = vec<R, 2>((R)0, (R)0);
    mac(d, a, b, zero);
}

template <typename T, int Cols = 8, int Rows = Cols, typename U>
inline mat<T> make_filled(U value) {
    mat<T> d;
    d.e = vec<T, 2>((T)value, (T)value);
    return d;
}

} // namespace ds4_sg

#define simdgroup_matrix              ds4_sg::mat
#define simdgroup_float8x8            ds4_sg::mat<float>
#define simdgroup_half8x8             ds4_sg::mat<half>
#define make_filled_simdgroup_matrix  ds4_sg::make_filled
#define simdgroup_load                ds4_sg::load
#define simdgroup_store               ds4_sg::store
#define simdgroup_multiply_accumulate ds4_sg::mac
#define simdgroup_multiply            ds4_sg::mul

#endif // DS4_METAL_SIMDGROUP_SHIM
