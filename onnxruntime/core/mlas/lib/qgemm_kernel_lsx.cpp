/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright (c) Vladislav Shchapov. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_lsx.cpp

Abstract:

    This module implements QGEMM kernels for LSX.

--*/

#include "mlasi.h"
#include "qgemm.h"
#include <lsxintrin.h>


#define LSX_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))


//
// Common functions
//

template<bool AIsSigned>
MLAS_FORCEINLINE
void
MlasGemmQuantCopyPackA_Impl(
    uint8_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const __m128i ZeroVector = __lsx_vldi(0);
    [[maybe_unused]] const __m128i BitFlipVector = __lsx_vreplgr2vr_w(0x80808080);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i ReductionVector = ZeroVector;

        //
        // Copy the source bytes to the packed buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 4 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //

        while (k >= 8) {

            __m128i Bytes = __lsx_vldrepl_d(& a[0], 0);

            if constexpr (AIsSigned)
            {
                Bytes = __lsx_vxor_v(Bytes, BitFlipVector);
            }

            __m128i Words = __lsx_vilvl_b(ZeroVector, Bytes);

            ReductionVector = __lsx_vadd_w(ReductionVector, __lsx_vhaddw_w_h(Words, Words));

            __lsx_vstelm_d(Bytes, & D[0], 0, 0);

            a += 8;
            D += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            if constexpr (AIsSigned)
            {
                __lsx_vstelm_d(BitFlipVector, & D[0], 0, 0);
            }
            else
            {
                __lsx_vstelm_d(ZeroVector, & D[0], 0, 0);
            }

            std::copy_n(&a[0], k, &D[0]);

            __m128i Bytes = __lsx_vldrepl_d(& D[0], 0);

            if constexpr (AIsSigned)
            {
                Bytes = __lsx_vxor_v(Bytes, BitFlipVector);
            }

            __m128i Words = __lsx_vilvl_b(ZeroVector, Bytes);

            ReductionVector = __lsx_vadd_w(ReductionVector, __lsx_vhaddw_w_h(Words, Words));

            __lsx_vstelm_d(Bytes, & D[0], 0, 0);

            D += (k + 3) & ~3;
        }

        //
        // Reduce the partial accumulators.
        //

        ReductionVector = __lsx_vadd_w(__lsx_vpickev_w(ReductionVector, ReductionVector), __lsx_vpickod_w(ReductionVector, ReductionVector));
        ReductionVector = __lsx_vadd_w(__lsx_vpickev_w(ReductionVector, ReductionVector), __lsx_vpickod_w(ReductionVector, ReductionVector));

        *RowSumBuffer++ = __lsx_vpickve2gr_w(ReductionVector, 0);

        A += lda;
        CountM -= 1;
    }
}



//
// U8X8 Kernel Implementations
//

struct MLAS_GEMM_U8X8_KERNEL_LSX
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 24, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 24, 128, 128 };
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_LSX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_LSX::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides;

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_LSX>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_LSX::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_LSX>(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    MlasGemmQuantCopyPackA_Impl<false>(D, A, lda, CountM, CountK, RowSumBuffer);
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessLsx(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* D,
    __m128i BytesRows[4],
    __m128i ColumnSums[2],
    __m128i BitFlipVector
)
{
    __m128i PairsInterleaved0 = __lsx_vilvl_b(BytesRows[1], BytesRows[0]);
    __m128i PairsInterleaved1 = __lsx_vilvl_b(BytesRows[3], BytesRows[2]);

    PairsInterleaved0 = __lsx_vxor_v(PairsInterleaved0, BitFlipVector);
    PairsInterleaved1 = __lsx_vxor_v(PairsInterleaved1, BitFlipVector);

    __m128i QuadsInterleaved0 = __lsx_vilvl_h(PairsInterleaved1, PairsInterleaved0);
    __m128i QuadsInterleaved1 = __lsx_vilvh_h(PairsInterleaved1, PairsInterleaved0);

    __m128i PairwiseAdd0 = __lsx_vhaddw_h_b(QuadsInterleaved0, QuadsInterleaved0);
    __m128i PairwiseAdd1 = __lsx_vhaddw_h_b(QuadsInterleaved1, QuadsInterleaved1);

    PairwiseAdd0 = __lsx_vhaddw_w_h(PairwiseAdd0, PairwiseAdd0);
    PairwiseAdd1 = __lsx_vhaddw_w_h(PairwiseAdd1, PairwiseAdd1);

    ColumnSums[0] = __lsx_vadd_w(ColumnSums[0], PairwiseAdd0);
    ColumnSums[1] = __lsx_vadd_w(ColumnSums[1], PairwiseAdd1);

    __lsx_vst(QuadsInterleaved0, & D[0], 0);
    __lsx_vst(QuadsInterleaved1, & D[16], 0);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_LSX>(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    __m128i BytesRows[4];

    const __m128i BitFlipVector = __lsx_vreplgr2vr_w(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = __lsx_vldi(0);
        ColumnSums[1] = __lsx_vldi(0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_LSX::PackedK) {

            BytesRows[0] = __lsx_vldrepl_d(& b[ldb * 0], 0);
            BytesRows[1] = __lsx_vldrepl_d(& b[ldb * 1], 0);
            BytesRows[2] = __lsx_vldrepl_d(& b[ldb * 2], 0);
            BytesRows[3] = __lsx_vldrepl_d(& b[ldb * 3], 0);

            MlasGemmU8X8CopyPackBProcessLsx(D, BytesRows, ColumnSums, BitFlipVector);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRows[0] = __lsx_vldrepl_d(& b[ldb * 0], 0);
            BytesRows[1] = BitFlipVector;
            BytesRows[2] = BitFlipVector;
            BytesRows[3] = BitFlipVector;

            if (k >= 2) {
                BytesRows[1] = __lsx_vldrepl_d(& b[ldb * 1], 0);
            }

            if (k >= 3) {
                BytesRows[2] = __lsx_vldrepl_d(& b[ldb * 2], 0);
            }

            MlasGemmU8X8CopyPackBProcessLsx(D, BytesRows, ColumnSums, BitFlipVector);

            D += 32;
        }

        __lsx_vst(ColumnSums[0], & ColumnSumBuffer[0], 0);
        __lsx_vst(ColumnSums[1], & ColumnSumBuffer[4], 0);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[32];

        ColumnSums[0] = __lsx_vldi(0);
        ColumnSums[1] = __lsx_vldi(0);

        while (CountK > 0) {

            size_t k = std::min(CountK, MLAS_GEMM_U8X8_KERNEL_LSX::PackedK);
            CountK -= k;

            __lsx_vst(BitFlipVector, & PaddedMatrixBData[0], 0);
            __lsx_vst(BitFlipVector, & PaddedMatrixBData[16], 0);

            uint8_t* padded = PaddedMatrixBData;

            do {

                std::copy_n(B, CountN, padded);

                padded += 8;
                B += ldb;
                k -= 1;

            } while (k > 0);

            BytesRows[0] = __lsx_vldrepl_d(& PaddedMatrixBData[0], 0);
            BytesRows[1] = __lsx_vldrepl_d(& PaddedMatrixBData[8], 0);
            BytesRows[2] = __lsx_vldrepl_d(& PaddedMatrixBData[16], 0);
            BytesRows[3] = __lsx_vldrepl_d(& PaddedMatrixBData[24], 0);

            MlasGemmU8X8CopyPackBProcessLsx(D, BytesRows, ColumnSums, BitFlipVector);

            D += 32;
        }

        __lsx_vst(ColumnSums[0], & ColumnSumBuffer[0], 0);
        __lsx_vst(ColumnSums[1], & ColumnSumBuffer[4], 0);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowLsx(
    __m128i ABroadcast,
    const MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* B,
    __m128i Accumulators[2]
)
{
    __m128i BElements0 = __lsx_vld(& B[0], 0);
    __m128i BElements1 = __lsx_vld(& B[16], 0);

    __m128i Intermediate0_0 = __lsx_vmulwev_h_bu_b(ABroadcast, BElements0);
    __m128i Intermediate0_1 = __lsx_vmulwod_h_bu_b(ABroadcast, BElements0);

    __m128i Intermediate1_0 = __lsx_vmulwev_h_bu_b(ABroadcast, BElements1);
    __m128i Intermediate1_1 = __lsx_vmulwod_h_bu_b(ABroadcast, BElements1);

    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vadd_w(__lsx_vhaddw_w_h(Intermediate0_0, Intermediate0_0), __lsx_vhaddw_w_h(Intermediate0_1, Intermediate0_1)));
    Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vadd_w(__lsx_vhaddw_w_h(Intermediate1_0, Intermediate1_0), __lsx_vhaddw_w_h(Intermediate1_1, Intermediate1_1)));
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_LSX>(
    const MLAS_GEMM_U8X8_KERNEL_LSX::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        Accumulators[0] = __lsx_vreplgr2vr_w(RowSumBuffer[0]);
        Accumulators[1] = Accumulators[0];

        if (ZeroPointB != nullptr) {
            Accumulators[0] = __lsx_vmul_w(Accumulators[0], __lsx_vld(& ZeroPointB[0], 0));
            Accumulators[1] = __lsx_vmul_w(Accumulators[1], __lsx_vld(& ZeroPointB[4], 0));
            ZeroPointB += 8;
        }

        Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld(& ColumnSumBuffer[0], 0));
        Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vld(& ColumnSumBuffer[4], 0));
        ColumnSumBuffer += 8;

        //
        // Broadcast each quad of 8-bit values from the matrix A and multiply
        // with the quad of 8-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const uint8_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = __lsx_vld(a, 0);
            __m128i ABroadcast;

            ABroadcast = __lsx_vshuf4i_w(AElements, LSX_SHUFFLE(0, 0, 0, 0));
            MlasGemmU8X8MultiplyAccumulateRowLsx(ABroadcast, &B[0], Accumulators);

            ABroadcast = __lsx_vshuf4i_w(AElements, LSX_SHUFFLE(1, 1, 1, 1));
            MlasGemmU8X8MultiplyAccumulateRowLsx(ABroadcast, &B[32], Accumulators);

            ABroadcast = __lsx_vshuf4i_w(AElements, LSX_SHUFFLE(2, 2, 2, 2));
            MlasGemmU8X8MultiplyAccumulateRowLsx(ABroadcast, &B[64], Accumulators);

            ABroadcast = __lsx_vshuf4i_w(AElements, LSX_SHUFFLE(3, 3, 3, 3));
            MlasGemmU8X8MultiplyAccumulateRowLsx(ABroadcast, &B[96], Accumulators);

            a += 4 * 4;
            B += 4 * 32;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = __lsx_vldrepl_w(a, 0);
            MlasGemmU8X8MultiplyAccumulateRowLsx(ABroadcast, &B[0], Accumulators);

            a += 4;
            B += 32;
            k -= 1;
        }

        //
        // Output the accumulator block after optionally accumulating the values
        // from matrix C.
        //

        if (CountN >= 8) {

            if (!ZeroMode) {
                Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld(& C[0], 0));
                Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vld(& C[4], 0));
            }

            __lsx_vst(Accumulators[0], & C[0], 0);
            __lsx_vst(Accumulators[1], & C[4], 0);

            C += 8;
            CountN -= 8;

        }
        else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld(& C[0], 0));
                }

                __lsx_vst(Accumulators[0], & C[0], 0);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vinsgr2vr_d(__lsx_vldrepl_d(& C[0], 0), 0, 1));
                }

                __lsx_vstelm_d( Accumulators[0], & C[0], 0, 0);
                C += 2;

                Accumulators[0] = __lsx_vshuf4i_w(Accumulators[0], LSX_SHUFFLE(3, 2, 3, 2));
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = __lsx_vpickve2gr_w(Accumulators[0], 0);

                if (!ZeroMode) {
                    AccumulatorValue += C[0];
                }

                C[0] = AccumulatorValue;
            }

            CountN = 0;
        }
    }

    return 1;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchLSX = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_LSX>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U8X8_KERNEL_LSX>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_LSX>,
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedK,
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.K,
    1  // assembly kernel M stride
};



//
// S8X8 Kernel Implementations
//

struct MLAS_GEMM_S8X8_KERNEL_LSX
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 24, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 24, 128, 128 };
};

constexpr size_t MLAS_GEMM_S8X8_KERNEL_LSX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8X8_KERNEL_LSX::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8X8_KERNEL_LSX::PackedStrides;


template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_S8X8_KERNEL_LSX>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_S8X8_KERNEL_LSX>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_S8X8_KERNEL_LSX::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_S8X8_KERNEL_LSX>(
    MLAS_GEMM_S8X8_KERNEL_LSX::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    MlasGemmQuantCopyPackA_Impl<true>(D, A, lda, CountM, CountK, RowSumBuffer);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_S8X8_KERNEL_LSX>(
    MLAS_GEMM_S8X8_KERNEL_LSX::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_LSX>(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_S8X8_KERNEL_LSX>(
    const MLAS_GEMM_S8X8_KERNEL_LSX::PackedAType* A,
    const MLAS_GEMM_S8X8_KERNEL_LSX::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_LSX>(A, B, C, PackedCountK, CountM, CountN, ldc, RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmS8X8DispatchLSX = {
    MlasGemmQuantOperation<MLAS_GEMM_S8X8_KERNEL_LSX>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_S8X8_KERNEL_LSX>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_S8X8_KERNEL_LSX>,
    MLAS_GEMM_S8X8_KERNEL_LSX::PackedK,
    MLAS_GEMM_S8X8_KERNEL_LSX::PackedStrides.K,
    1  // assembly kernel M stride
};


static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::PackedK == MLAS_GEMM_S8X8_KERNEL_LSX::PackedK);

static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::Strides.M == MLAS_GEMM_U8X8_KERNEL_LSX::Strides.M);
static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::Strides.N == MLAS_GEMM_U8X8_KERNEL_LSX::Strides.N);
static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::Strides.K == MLAS_GEMM_U8X8_KERNEL_LSX::Strides.K);

static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.M == MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.M);
static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.N == MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.N);
static_assert(MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.K == MLAS_GEMM_U8X8_KERNEL_LSX::PackedStrides.K);
