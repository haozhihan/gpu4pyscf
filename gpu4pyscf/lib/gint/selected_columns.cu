/*
 * Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Selected shell-pair ERI columns.
 *
 * One CUDA thread evaluates one shell quartet and immediately contracts its
 * Cartesian result into original spherical AO-pair coefficients.  Integral
 * values are written directly to a batch-by-packed-pair FP64 output.  The
 * implementation does not allocate an AO^4 tensor, a pair-square matrix, or
 * a per-pivot AO matrix.
 *
 * This correctness-first kernel intentionally uses the existing generic Rys
 * primitive.  Performance reporting is controlled by the Python provider's
 * separately issued, source-and-binary-bound water2 qualification receipt.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "g2e.h"
#include "cint2e.cuh"
#include "rys_roots.cu"

/*
 * The generic GINT one-root specialization evaluates erf(sqrt(x))/sqrt(x)
 * directly and is singular at x == 0.  The production root-1 kernels use the
 * same small-x limiting values below.  Route only this translation unit's
 * generic G2E primitive through a safe wrapper while reusing all other Rys
 * implementations unchanged.
 */
template <int NROOTS> __device__ __forceinline__
void GINTselected_rys_root(double x, double *rw)
{
    GINTrys_root<NROOTS>(x, rw);
}

template <> __device__ __forceinline__
void GINTselected_rys_root<1>(double x, double *rw)
{
    if (x < 3.e-7) {
        rw[0] = .5;
        rw[1] = 1.;
        return;
    }
    GINTrys_root<1>(x, rw);
}

__device__ __forceinline__
void GINTselected_rys_root(int nroots, double x, double *rw)
{
    GINTrys_root(nroots, x, rw);
}

#define GINTrys_root GINTselected_rys_root
#include "g2e.cu"
#undef GINTrys_root
#include "gout2e.cuh"

enum {
    GINT_SELECTED_SUCCESS = 0,
    GINT_SELECTED_INVALID_ARGUMENT = 1,
    GINT_SELECTED_UNSUPPORTED_RYS_ORDER = 2,
    GINT_SELECTED_GRID_OVERFLOW = 3,
    GINT_SELECTED_CONSTANT_COPY_FAILED = 4,
    GINT_SELECTED_KERNEL_LAUNCH_FAILED = 5
};

enum {
    GINT_SELECTED_ROW_GROUP_DIAGONAL = 1,
    GINT_SELECTED_GROUPED_COLUMNS = 2,
    GINT_SELECTED_COLUMN_FLAG_MASK = 3
};

/*
 * Rys-7/8 instantiate GOUTSIZE7/8 as 21,720/45,750 doubles.  Keeping that
 * array automatic gives each thread a 177,472/369,600-byte static stack on
 * sm_80.  CUDA backs local memory for the context-wide maximum thread
 * envelope, independent of this kernel's achieved occupancy, so merely
 * reducing occupancy does not bound HBM.  The high-order kernels below use
 * one explicit, provider-owned global scratch slice per thread instead.
 * A single-warp block and grid-stride loop bound the number of live slices.
 */
enum {
    GINT_SELECTED_HIGH_RYS_MIN = 7,
    GINT_SELECTED_HIGH_RYS_THREADS = 32
};

__device__ __forceinline__
double selected_coeff(const GINTSelectedPairData data, int cart, int original)
{
    return data.coeff[(size_t)cart * data.nao_original + original];
}

__device__ __forceinline__
double selected_pair_weight(const GINTSelectedPairData data,
                            int cart_first, int cart_second,
                            int original_first, int original_second,
                            int symmetrize)
{
    double value = selected_coeff(data, cart_first, original_first)
                 * selected_coeff(data, cart_second, original_second);
    if (symmetrize) {
        value += selected_coeff(data, cart_first, original_second)
               * selected_coeff(data, cart_second, original_first);
    }
    return value;
}

template <int NROOTS, int GOUTSIZE> __device__ __forceinline__
void selected_build_gout(const GINTEnvVars envs,
                         int bas_ij, int bas_kl,
                         int prim_ij, int prim_kl,
                         double *gout)
{
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double *g = gout + envs.nf;
    for (int element = 0; element < envs.nf; ++element) {
        gout[element] = 0.;
    }
    for (int ij = prim_ij; ij < prim_ij + envs.nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl + envs.nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(envs, g, envs.fac,
                                   ish, jsh, ksh, lsh, ij, kl);
            GINTgout2e<NROOTS>(envs, gout, g);
        }
    }
}

__device__ __forceinline__
void selected_contract_column(const GINTEnvVars envs,
                              const GINTSelectedPairData data,
                              double *columns,
                              int batch_size,
                              int pair,
                              int output_row,
                              int cp_kl_id,
                              int row_group_diagonal,
                              int bas_ij,
                              int bas_kl,
                              const double *gout)
{
    if (pair < 0 || pair >= data.npair ||
        output_row < 0 || output_row >= batch_size ||
        data.pair_cp_id[pair] != cp_kl_id) {
        return;
    }
    int ish = c_bpcache.bas_pair2bra[bas_ij];
    int jsh = c_bpcache.bas_pair2ket[bas_ij];
    int ksh = c_bpcache.bas_pair2bra[bas_kl];
    int lsh = c_bpcache.bas_pair2ket[bas_kl];
    int pivot_first = data.pair_i[pair];
    int pivot_second = data.pair_j[pair];
    int pivot_symmetrize = (int)data.pair_symmetrize[pair];
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int original_i0 = data.shell_original_offsets[ish];
    int original_i1 = data.shell_original_offsets[ish + 1];
    int original_j0 = data.shell_original_offsets[jsh];
    int original_j1 = data.shell_original_offsets[jsh + 1];
    double right_weights[GPU_AO_NF * GPU_AO_NF];
    for (int lc = 0; lc < envs.nfl; ++lc) {
        for (int kc = 0; kc < envs.nfk; ++kc) {
            right_weights[lc * envs.nfk + kc] = selected_pair_weight(
                data, k0 + kc, l0 + lc,
                pivot_first, pivot_second, pivot_symmetrize);
        }
    }

    for (int pi = original_i0; pi < original_i1; ++pi) {
        int original_first = data.shell_original_aos[pi];
        for (int pj = original_j0; pj < original_j1; ++pj) {
            int original_second = data.shell_original_aos[pj];
            if (row_group_diagonal && original_first < original_second) {
                continue;
            }
            int high = original_first > original_second
                     ? original_first : original_second;
            int low = original_first > original_second
                    ? original_second : original_first;
            size_t packed = (size_t)high * (high + 1) / 2 + low;
            double left_weights[GPU_AO_NF * GPU_AO_NF];
            for (int jc = 0; jc < envs.nfj; ++jc) {
                for (int ic = 0; ic < envs.nfi; ++ic) {
                    left_weights[jc * envs.nfi + ic] = selected_pair_weight(
                        data, i0 + ic, j0 + jc,
                        original_first, original_second,
                        !row_group_diagonal);
                }
            }
            double value = 0.;
            for (int lc = 0; lc < envs.nfl; ++lc) {
                for (int kc = 0; kc < envs.nfk; ++kc) {
                    double right = right_weights[lc * envs.nfk + kc];
                    if (right == 0.) {
                        continue;
                    }
                    for (int jc = 0; jc < envs.nfj; ++jc) {
                        for (int ic = 0; ic < envs.nfi; ++ic) {
                            double left = left_weights[jc * envs.nfi + ic];
                            size_t gout_index =
                                (((size_t)lc * envs.nfk + kc) * envs.nfj + jc)
                                * envs.nfi + ic;
                            value += gout[gout_index] * left * right;
                        }
                    }
                }
            }
            columns[(size_t)output_row * data.npair + packed] = value;
        }
    }
}

template <int NROOTS, int GOUTSIZE> __device__ __forceinline__
void selected_columns_task(GINTEnvVars envs,
                           GINTSelectedPairData data,
                           double *columns,
                           int batch_size,
                           const int *selected_pairs,
                           const int *selected_rows,
                           int selected_count,
                           int cp_ij_id,
                           int cp_kl_id,
                           int row_group_diagonal,
                           BasisProdOffsets offsets,
                           double log_cutoff,
                           size_t linear,
                           double *gout)
{
    int task_ij = (int)(linear % offsets.ntasks_ij);
    int selected = (int)(linear / offsets.ntasks_ij);
    int pair = selected_pairs[selected];
    int output_row = selected_rows[selected];
    if (pair < 0 || pair >= data.npair ||
        output_row < 0 || output_row >= batch_size ||
        data.pair_cp_id[pair] != cp_kl_id) {
        return;
    }
    int task_kl = data.pair_task_id[pair];
    if (task_kl < 0 || task_kl >= offsets.ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < 0 || bas_ij >= data.task_count ||
        bas_kl < 0 || bas_kl >= data.task_count) {
        return;
    }
    if (data.task_log_q[bas_ij] + data.task_log_q[bas_kl] < log_cutoff) {
        return;
    }
    int prim_ij = offsets.primitive_ij + task_ij * envs.nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * envs.nprim_kl;
    selected_build_gout<NROOTS, GOUTSIZE>(
        envs, bas_ij, bas_kl, prim_ij, prim_kl, gout);
    selected_contract_column(
        envs, data, columns, batch_size, pair, output_row, cp_kl_id,
        row_group_diagonal, bas_ij, bas_kl, gout);
}

template <int NROOTS, int GOUTSIZE> __device__ __forceinline__
void selected_columns_grouped_task(GINTEnvVars envs,
                                   GINTSelectedPairData data,
                                   double *columns,
                                   int batch_size,
                                   const int *selected_pairs,
                                   const int *selected_rows,
                                   int selected_count,
                                   int cp_ij_id,
                                   int cp_kl_id,
                                   int row_group_diagonal,
                                   BasisProdOffsets offsets,
                                   double log_cutoff,
                                   size_t linear,
                                   double *gout)
{
    int task_ij = (int)(linear % offsets.ntasks_ij);
    int selected = (int)(linear / offsets.ntasks_ij);
    int pair = selected_pairs[selected];
    if (pair < 0 || pair >= data.npair ||
        data.pair_cp_id[pair] != cp_kl_id) {
        return;
    }
    int task_kl = data.pair_task_id[pair];
    if (task_kl < 0 || task_kl >= offsets.ntasks_kl) {
        return;
    }

    /*
     * The Python provider stable-sorts each cp-local request by ket task.
     * Only the first pivot in a contiguous task group evaluates GOUT; this
     * thread then contracts all pivots in the group into their original rows.
     * The validity checks keep an unsorted or malformed private-ABI caller
     * correct and bounded, though such a caller simply gets less reuse.
     */
    if (selected > 0) {
        int previous_pair = selected_pairs[selected - 1];
        if (previous_pair >= 0 && previous_pair < data.npair &&
            data.pair_cp_id[previous_pair] == cp_kl_id &&
            data.pair_task_id[previous_pair] == task_kl) {
            return;
        }
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < 0 || bas_ij >= data.task_count ||
        bas_kl < 0 || bas_kl >= data.task_count) {
        return;
    }
    if (data.task_log_q[bas_ij] + data.task_log_q[bas_kl] < log_cutoff) {
        return;
    }
    int prim_ij = offsets.primitive_ij + task_ij * envs.nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * envs.nprim_kl;
    selected_build_gout<NROOTS, GOUTSIZE>(
        envs, bas_ij, bas_kl, prim_ij, prim_kl, gout);

    for (int member = selected; member < selected_count; ++member) {
        int member_pair = selected_pairs[member];
        if (member_pair < 0 || member_pair >= data.npair ||
            data.pair_cp_id[member_pair] != cp_kl_id ||
            data.pair_task_id[member_pair] != task_kl) {
            break;
        }
        selected_contract_column(
            envs, data, columns, batch_size, member_pair,
            selected_rows[member], cp_kl_id, row_group_diagonal,
            bas_ij, bas_kl, gout);
    }
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_columns_kernel_cutoff(GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *columns,
                                    int batch_size,
                                    const int *selected_pairs,
                                    const int *selected_rows,
                                    int selected_count,
                                    int cp_ij_id,
                                    int cp_kl_id,
                                    int row_group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff)
{
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    if (linear >= work) {
        return;
    }
    double gout[GOUTSIZE];
    selected_columns_task<NROOTS, GOUTSIZE>(
        envs, data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, offsets,
        log_cutoff, linear, gout);
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_columns_kernel_grouped_cutoff(
                                    GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *columns,
                                    int batch_size,
                                    const int *selected_pairs,
                                    const int *selected_rows,
                                    int selected_count,
                                    int cp_ij_id,
                                    int cp_kl_id,
                                    int row_group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff)
{
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    if (linear >= work) {
        return;
    }
    double gout[GOUTSIZE];
    selected_columns_grouped_task<NROOTS, GOUTSIZE>(
        envs, data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, offsets,
        log_cutoff, linear, gout);
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_columns_kernel_bounded_workspace(
                                    GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *columns,
                                    int batch_size,
                                    const int *selected_pairs,
                                    const int *selected_rows,
                                    int selected_count,
                                    int cp_ij_id,
                                    int cp_kl_id,
                                    int row_group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff,
                                    double *workspace)
{
    size_t slot = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    double *gout = workspace + slot * GOUTSIZE;
    for (size_t linear = slot; linear < work; linear += stride) {
        selected_columns_task<NROOTS, GOUTSIZE>(
            envs, data, columns, batch_size, selected_pairs, selected_rows,
            selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, offsets,
            log_cutoff, linear, gout);
    }
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_columns_kernel_grouped_bounded_workspace(
                                    GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *columns,
                                    int batch_size,
                                    const int *selected_pairs,
                                    const int *selected_rows,
                                    int selected_count,
                                    int cp_ij_id,
                                    int cp_kl_id,
                                    int row_group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff,
                                    double *workspace)
{
    size_t slot = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    double *gout = workspace + slot * GOUTSIZE;
    for (size_t linear = slot; linear < work; linear += stride) {
        selected_columns_grouped_task<NROOTS, GOUTSIZE>(
            envs, data, columns, batch_size, selected_pairs, selected_rows,
            selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, offsets,
            log_cutoff, linear, gout);
    }
}

template <int NROOTS, int GOUTSIZE> __device__ __forceinline__
void selected_diagonal_task(GINTEnvVars envs,
                            GINTSelectedPairData data,
                            double *diagonal,
                            const int *selected_pairs,
                            int selected_count,
                            int cp_id,
                            int group_diagonal,
                            BasisProdOffsets offsets,
                            double log_cutoff,
                            int selected,
                            double *gout)
{
    int pair = selected_pairs[selected];
    if (pair < 0 || pair >= data.npair || data.pair_cp_id[pair] != cp_id) {
        return;
    }
    int task = data.pair_task_id[pair];
    int symmetrize = (int)data.pair_symmetrize[pair];
    if (task < 0 || task >= offsets.ntasks_ij ||
        symmetrize == group_diagonal) {
        return;
    }
    int bas_pair = offsets.bas_ij + task;
    if (bas_pair < 0 || bas_pair >= data.task_count) {
        return;
    }
    if (2. * data.task_log_q[bas_pair] < log_cutoff) {
        return;
    }
    int primitive = offsets.primitive_ij + task * envs.nprim_ij;
    int ish = c_bpcache.bas_pair2bra[bas_pair];
    int jsh = c_bpcache.bas_pair2ket[bas_pair];
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int original_first = data.pair_i[pair];
    int original_second = data.pair_j[pair];

    selected_build_gout<NROOTS, GOUTSIZE>(
        envs, bas_pair, bas_pair, primitive, primitive, gout);
    double weights[GPU_AO_NF * GPU_AO_NF];
    for (int jc = 0; jc < envs.nfj; ++jc) {
        for (int ic = 0; ic < envs.nfi; ++ic) {
            weights[jc * envs.nfi + ic] = selected_pair_weight(
                data, i0 + ic, j0 + jc,
                original_first, original_second, symmetrize);
        }
    }
    double value = 0.;
    for (int lc = 0; lc < envs.nfl; ++lc) {
        for (int kc = 0; kc < envs.nfk; ++kc) {
            double right = weights[lc * envs.nfk + kc];
            if (right == 0.) {
                continue;
            }
            for (int jc = 0; jc < envs.nfj; ++jc) {
                for (int ic = 0; ic < envs.nfi; ++ic) {
                    double left = weights[jc * envs.nfi + ic];
                    size_t gout_index =
                        (((size_t)lc * envs.nfk + kc) * envs.nfj + jc)
                        * envs.nfi + ic;
                    value += gout[gout_index] * left * right;
                }
            }
        }
    }
    diagonal[pair] = value;
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_diagonal_kernel(GINTEnvVars envs,
                              GINTSelectedPairData data,
                              double *diagonal,
                              const int *selected_pairs,
                              int selected_count,
                              int cp_id,
                              int group_diagonal,
                              BasisProdOffsets offsets,
                              double log_cutoff)
{
    int selected = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (selected >= selected_count) {
        return;
    }
    double gout[GOUTSIZE];
    selected_diagonal_task<NROOTS, GOUTSIZE>(
        envs, data, diagonal, selected_pairs, selected_count, cp_id,
        group_diagonal, offsets, log_cutoff, selected, gout);
}

template <int NROOTS, int GOUTSIZE> __global__
void selected_diagonal_kernel_bounded_workspace(
                              GINTEnvVars envs,
                              GINTSelectedPairData data,
                              double *diagonal,
                              const int *selected_pairs,
                              int selected_count,
                              int cp_id,
                              int group_diagonal,
                              BasisProdOffsets offsets,
                              double log_cutoff,
                              double *workspace)
{
    size_t slot = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double *gout = workspace + slot * GOUTSIZE;
    for (size_t selected = slot; selected < (size_t)selected_count;
         selected += stride) {
        selected_diagonal_task<NROOTS, GOUTSIZE>(
            envs, data, diagonal, selected_pairs, selected_count, cp_id,
            group_diagonal, offsets, log_cutoff, (int)selected, gout);
    }
}

template <int NROOTS, int GOUTSIZE>
static int launch_selected_columns(cudaStream_t stream,
                                   GINTEnvVars envs,
                                   GINTSelectedPairData data,
                                   double *columns,
                                   int batch_size,
                                   const int *selected_pairs,
                                   const int *selected_rows,
                                   int selected_count,
                                   int cp_ij_id,
                                   int cp_kl_id,
                                   int row_group_diagonal,
                                   int grouped,
                                   BasisProdOffsets offsets,
                                   double log_cutoff)
{
    const unsigned int threads = 64;
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    size_t blocks = (work + threads - 1) / threads;
    if (blocks > 2147483647ULL) {
        return GINT_SELECTED_GRID_OVERFLOW;
    }
    if (grouped) {
        selected_columns_kernel_grouped_cutoff<NROOTS, GOUTSIZE>
            <<<dim3((unsigned int)blocks), dim3(threads), 0, stream>>>(
                envs, data, columns, batch_size, selected_pairs, selected_rows,
                selected_count, cp_ij_id, cp_kl_id, row_group_diagonal,
                offsets, log_cutoff);
    } else {
        selected_columns_kernel_cutoff<NROOTS, GOUTSIZE>
            <<<dim3((unsigned int)blocks), dim3(threads), 0, stream>>>(
                envs, data, columns, batch_size, selected_pairs, selected_rows,
                selected_count, cp_ij_id, cp_kl_id, row_group_diagonal,
                offsets, log_cutoff);
    }
    return cudaGetLastError() == cudaSuccess
         ? GINT_SELECTED_SUCCESS : GINT_SELECTED_KERNEL_LAUNCH_FAILED;
}

template <int NROOTS, int GOUTSIZE>
static int launch_selected_diagonal(cudaStream_t stream,
                                    GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *diagonal,
                                    const int *selected_pairs,
                                    int selected_count,
                                    int cp_id,
                                    int group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff)
{
    const unsigned int threads = 64;
    size_t blocks = ((size_t)selected_count + threads - 1) / threads;
    if (blocks > 2147483647ULL) {
        return GINT_SELECTED_GRID_OVERFLOW;
    }
    selected_diagonal_kernel<NROOTS, GOUTSIZE>
        <<<dim3((unsigned int)blocks), dim3(threads), 0, stream>>>(
            envs, data, diagonal, selected_pairs, selected_count, cp_id,
            group_diagonal, offsets, log_cutoff);
    return cudaGetLastError() == cudaSuccess
         ? GINT_SELECTED_SUCCESS : GINT_SELECTED_KERNEL_LAUNCH_FAILED;
}

template <int NROOTS, int GOUTSIZE>
static int launch_selected_columns_bounded_workspace(
                                   cudaStream_t stream,
                                   GINTEnvVars envs,
                                   GINTSelectedPairData data,
                                   double *columns,
                                   int batch_size,
                                   const int *selected_pairs,
                                   const int *selected_rows,
                                   int selected_count,
                                   int cp_ij_id,
                                   int cp_kl_id,
                                   int row_group_diagonal,
                                   int grouped,
                                   BasisProdOffsets offsets,
                                   double log_cutoff,
                                   void *workspace,
                                   size_t workspace_bytes)
{
    const size_t threads = GINT_SELECTED_HIGH_RYS_THREADS;
    const size_t bytes_per_block = threads * GOUTSIZE * sizeof(double);
    if (workspace == NULL || workspace_bytes < bytes_per_block) {
        return GINT_SELECTED_INVALID_ARGUMENT;
    }
    size_t work = (size_t)offsets.ntasks_ij * selected_count;
    size_t blocks = (work + threads - 1) / threads;
    size_t workspace_blocks = workspace_bytes / bytes_per_block;
    if (blocks > workspace_blocks) {
        blocks = workspace_blocks;
    }
    if (blocks < 1 || blocks > 2147483647ULL) {
        return blocks < 1
             ? GINT_SELECTED_INVALID_ARGUMENT : GINT_SELECTED_GRID_OVERFLOW;
    }
    if (grouped) {
        selected_columns_kernel_grouped_bounded_workspace<NROOTS, GOUTSIZE>
            <<<dim3((unsigned int)blocks),
               dim3((unsigned int)threads), 0, stream>>>(
                envs, data, columns, batch_size,
                selected_pairs, selected_rows, selected_count,
                cp_ij_id, cp_kl_id, row_group_diagonal, offsets, log_cutoff,
                static_cast<double *>(workspace));
    } else {
        selected_columns_kernel_bounded_workspace<NROOTS, GOUTSIZE>
            <<<dim3((unsigned int)blocks),
               dim3((unsigned int)threads), 0, stream>>>(
                envs, data, columns, batch_size,
                selected_pairs, selected_rows, selected_count,
                cp_ij_id, cp_kl_id, row_group_diagonal, offsets, log_cutoff,
                static_cast<double *>(workspace));
    }
    return cudaGetLastError() == cudaSuccess
         ? GINT_SELECTED_SUCCESS : GINT_SELECTED_KERNEL_LAUNCH_FAILED;
}

template <int NROOTS, int GOUTSIZE>
static int launch_selected_diagonal_bounded_workspace(
                                    cudaStream_t stream,
                                    GINTEnvVars envs,
                                    GINTSelectedPairData data,
                                    double *diagonal,
                                    const int *selected_pairs,
                                    int selected_count,
                                    int cp_id,
                                    int group_diagonal,
                                    BasisProdOffsets offsets,
                                    double log_cutoff,
                                    void *workspace,
                                    size_t workspace_bytes)
{
    const size_t threads = GINT_SELECTED_HIGH_RYS_THREADS;
    const size_t bytes_per_block = threads * GOUTSIZE * sizeof(double);
    if (workspace == NULL || workspace_bytes < bytes_per_block) {
        return GINT_SELECTED_INVALID_ARGUMENT;
    }
    size_t work = (size_t)selected_count;
    size_t blocks = (work + threads - 1) / threads;
    size_t workspace_blocks = workspace_bytes / bytes_per_block;
    if (blocks > workspace_blocks) {
        blocks = workspace_blocks;
    }
    if (blocks < 1 || blocks > 2147483647ULL) {
        return blocks < 1
             ? GINT_SELECTED_INVALID_ARGUMENT : GINT_SELECTED_GRID_OVERFLOW;
    }
    selected_diagonal_kernel_bounded_workspace<NROOTS, GOUTSIZE>
        <<<dim3((unsigned int)blocks),
           dim3((unsigned int)threads), 0, stream>>>(
            envs, data, diagonal, selected_pairs, selected_count, cp_id,
            group_diagonal, offsets, log_cutoff,
            static_cast<double *>(workspace));
    return cudaGetLastError() == cudaSuccess
         ? GINT_SELECTED_SUCCESS : GINT_SELECTED_KERNEL_LAUNCH_FAILED;
}

static size_t selected_high_rys_gout_size(int maximum_rys_order)
{
    switch (maximum_rys_order) {
    case 7: return GOUTSIZE7;
    case 8: return GOUTSIZE8;
    default: return 0;
    }
}

static size_t selected_workspace_size(int maximum_rys_order,
                                      int multiprocessor_count)
{
    size_t gout_size = selected_high_rys_gout_size(maximum_rys_order);
    if (gout_size == 0 || multiprocessor_count < 1) {
        return 0;
    }
    size_t factor = (size_t)GINT_SELECTED_HIGH_RYS_THREADS
                  * (size_t)multiprocessor_count;
    if (gout_size > SIZE_MAX / sizeof(double) / factor) {
        return 0;
    }
    return gout_size * sizeof(double) * factor;
}

static int validate_selected_arguments(BasisProdCache *bpcache,
                                       const GINTSelectedPairData *data,
                                       int cp_first, int cp_second,
                                       double log_cutoff, double omega)
{
    if (bpcache == NULL || data == NULL || data->npair < 1 ||
        data->nao_original < 1 || data->nbas != bpcache->nbas ||
        data->spherical != 1 || data->single_shell_support != 1 ||
        data->abi_version != 2 || data->coeff_rows < data->nao_original ||
        data->shell_original_aos_count != data->nao_original ||
        data->task_count != bpcache->bas_pairs_locs[bpcache->ncptype] ||
        data->coeff == NULL || data->shell_original_offsets == NULL ||
        data->shell_original_aos == NULL || data->pair_i == NULL ||
        data->pair_j == NULL || data->pair_cp_id == NULL ||
        data->pair_task_id == NULL || data->pair_symmetrize == NULL ||
        data->task_log_q == NULL || !isfinite(log_cutoff) || !isfinite(omega) ||
        cp_first < 0 || cp_first >= bpcache->ncptype ||
        cp_second < 0 || cp_second >= bpcache->ncptype) {
        return GINT_SELECTED_INVALID_ARGUMENT;
    }
    return GINT_SELECTED_SUCCESS;
}

static int prepare_selected_env(BasisProdCache *bpcache,
                                int cp_ij_id, int cp_kl_id,
                                double omega, GINTEnvVars *envs)
{
    int ng[4] = {0, 0, 0, 0};
    GINTinit_EnvVars(envs, bpcache->cptype + cp_ij_id,
                    bpcache->cptype + cp_kl_id, ng);
    envs->omega = omega;
    if (envs->nrys_roots < 1 || envs->nrys_roots > 8) {
        return GINT_SELECTED_UNSUPPORTED_RYS_ORDER;
    }
    return GINT_SELECTED_SUCCESS;
}

extern "C" {

int GINTfill_selected_int2e_columns(
    void *stream_pointer, BasisProdCache *bpcache,
    const GINTSelectedPairData *data, double *columns, int batch_size,
    const int *selected_pairs, const int *selected_rows, int selected_count,
    int cp_ij_id, int cp_kl_id, int row_group_diagonal,
    double log_cutoff, double omega, void *workspace, size_t workspace_bytes,
    int *constant_copy_performed)
{
    if (constant_copy_performed == NULL) {
        return GINT_SELECTED_INVALID_ARGUMENT;
    }
    *constant_copy_performed = 0;
    int error = validate_selected_arguments(
        bpcache, data, cp_ij_id, cp_kl_id, log_cutoff, omega);
    if (error != GINT_SELECTED_SUCCESS || columns == NULL || batch_size < 1 ||
        selected_pairs == NULL || selected_rows == NULL || selected_count < 1 ||
        selected_count > batch_size || row_group_diagonal < 0 ||
        (row_group_diagonal & ~GINT_SELECTED_COLUMN_FLAG_MASK) != 0) {
        return error == GINT_SELECTED_SUCCESS
             ? GINT_SELECTED_INVALID_ARGUMENT : error;
    }
    if (bpcache->cptype[cp_ij_id].npairs < 1 ||
        bpcache->cptype[cp_kl_id].npairs < 1) {
        return GINT_SELECTED_SUCCESS;
    }
    int grouped = row_group_diagonal & GINT_SELECTED_GROUPED_COLUMNS;
    row_group_diagonal &= GINT_SELECTED_ROW_GROUP_DIAGONAL;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    cudaError_t copy_error = cudaMemcpyToSymbolAsync(
        c_bpcache, bpcache, sizeof(BasisProdCache), 0,
        cudaMemcpyHostToDevice, stream);
    if (copy_error != cudaSuccess) {
        return GINT_SELECTED_CONSTANT_COPY_FAILED;
    }
    *constant_copy_performed = 1;
    GINTEnvVars envs;
    error = prepare_selected_env(
        bpcache, cp_ij_id, cp_kl_id, omega, &envs);
    if (error != GINT_SELECTED_SUCCESS) {
        return error;
    }
    BasisProdOffsets offsets;
    offsets.ntasks_ij = bpcache->cptype[cp_ij_id].npairs;
    offsets.ntasks_kl = bpcache->cptype[cp_kl_id].npairs;
    offsets.bas_ij = bpcache->bas_pairs_locs[cp_ij_id];
    offsets.bas_kl = bpcache->bas_pairs_locs[cp_kl_id];
    offsets.primitive_ij = bpcache->primitive_pairs_locs[cp_ij_id];
    offsets.primitive_kl = bpcache->primitive_pairs_locs[cp_kl_id];
    switch (envs.nrys_roots) {
    case 1: return launch_selected_columns<1, GOUTSIZE1>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 2: return launch_selected_columns<2, GOUTSIZE2>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 3: return launch_selected_columns<3, GOUTSIZE3>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 4: return launch_selected_columns<4, GOUTSIZE4>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 5: return launch_selected_columns<5, GOUTSIZE5>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 6: return launch_selected_columns<6, GOUTSIZE6>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff);
    case 7: return launch_selected_columns_bounded_workspace<7, GOUTSIZE7>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff, workspace, workspace_bytes);
    case 8: return launch_selected_columns_bounded_workspace<8, GOUTSIZE8>(
        stream, envs, *data, columns, batch_size, selected_pairs, selected_rows,
        selected_count, cp_ij_id, cp_kl_id, row_group_diagonal, grouped, offsets,
        log_cutoff, workspace, workspace_bytes);
    default: return GINT_SELECTED_UNSUPPORTED_RYS_ORDER;
    }
}

int GINTfill_selected_int2e_diagonal(
    void *stream_pointer, BasisProdCache *bpcache,
    const GINTSelectedPairData *data, double *diagonal,
    const int *selected_pairs, int selected_count, int cp_id,
    int group_diagonal, double log_cutoff, double omega,
    void *workspace, size_t workspace_bytes,
    int *constant_copy_performed)
{
    if (constant_copy_performed == NULL) {
        return GINT_SELECTED_INVALID_ARGUMENT;
    }
    *constant_copy_performed = 0;
    int error = validate_selected_arguments(
        bpcache, data, cp_id, cp_id, log_cutoff, omega);
    if (error != GINT_SELECTED_SUCCESS || diagonal == NULL ||
        selected_pairs == NULL || selected_count < 1 ||
        (group_diagonal != 0 && group_diagonal != 1)) {
        return error == GINT_SELECTED_SUCCESS
             ? GINT_SELECTED_INVALID_ARGUMENT : error;
    }
    if (bpcache->cptype[cp_id].npairs < 1) {
        return GINT_SELECTED_SUCCESS;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    cudaError_t copy_error = cudaMemcpyToSymbolAsync(
        c_bpcache, bpcache, sizeof(BasisProdCache), 0,
        cudaMemcpyHostToDevice, stream);
    if (copy_error != cudaSuccess) {
        return GINT_SELECTED_CONSTANT_COPY_FAILED;
    }
    *constant_copy_performed = 1;
    GINTEnvVars envs;
    error = prepare_selected_env(bpcache, cp_id, cp_id, omega, &envs);
    if (error != GINT_SELECTED_SUCCESS) {
        return error;
    }
    BasisProdOffsets offsets;
    offsets.ntasks_ij = bpcache->cptype[cp_id].npairs;
    offsets.ntasks_kl = offsets.ntasks_ij;
    offsets.bas_ij = bpcache->bas_pairs_locs[cp_id];
    offsets.bas_kl = offsets.bas_ij;
    offsets.primitive_ij = bpcache->primitive_pairs_locs[cp_id];
    offsets.primitive_kl = offsets.primitive_ij;
    switch (envs.nrys_roots) {
    case 1: return launch_selected_diagonal<1, GOUTSIZE1>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 2: return launch_selected_diagonal<2, GOUTSIZE2>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 3: return launch_selected_diagonal<3, GOUTSIZE3>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 4: return launch_selected_diagonal<4, GOUTSIZE4>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 5: return launch_selected_diagonal<5, GOUTSIZE5>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 6: return launch_selected_diagonal<6, GOUTSIZE6>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff);
    case 7: return launch_selected_diagonal_bounded_workspace<7, GOUTSIZE7>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff,
        workspace, workspace_bytes);
    case 8: return launch_selected_diagonal_bounded_workspace<8, GOUTSIZE8>(
        stream, envs, *data, diagonal, selected_pairs, selected_count,
        cp_id, group_diagonal, offsets, log_cutoff,
        workspace, workspace_bytes);
    default: return GINT_SELECTED_UNSUPPORTED_RYS_ORDER;
    }
}

}  /* extern "C" */
extern "C" {
size_t GINTselected_workspace_size(int maximum_rys_order,
                                   int multiprocessor_count)
{
    return selected_workspace_size(maximum_rys_order, multiprocessor_count);
}

size_t GINTsizeof_selected_pair_data(void)
{
    return sizeof(GINTSelectedPairData);
}

size_t GINToffsetof_selected_pair_data(int field_index)
{
    static const size_t offsets[] = {
        offsetof(GINTSelectedPairData, npair),
        offsetof(GINTSelectedPairData, nao_original),
        offsetof(GINTSelectedPairData, nbas),
        offsetof(GINTSelectedPairData, spherical),
        offsetof(GINTSelectedPairData, single_shell_support),
        offsetof(GINTSelectedPairData, abi_version),
        offsetof(GINTSelectedPairData, coeff_rows),
        offsetof(GINTSelectedPairData, shell_original_aos_count),
        offsetof(GINTSelectedPairData, task_count),
        offsetof(GINTSelectedPairData, coeff),
        offsetof(GINTSelectedPairData, shell_original_offsets),
        offsetof(GINTSelectedPairData, shell_original_aos),
        offsetof(GINTSelectedPairData, pair_i),
        offsetof(GINTSelectedPairData, pair_j),
        offsetof(GINTSelectedPairData, pair_cp_id),
        offsetof(GINTSelectedPairData, pair_task_id),
        offsetof(GINTSelectedPairData, pair_symmetrize),
        offsetof(GINTSelectedPairData, task_log_q),
    };
    const int count = (int)(sizeof(offsets) / sizeof(offsets[0]));
    return field_index >= 0 && field_index < count
        ? offsets[field_index] : (size_t)-1;
}

int GINTselected_pair_data_abi_version(void)
{
    return 2;
}
}
