# THC-RR direct complement implementation contract

This note fixes the algebraic boundary for replacing the validation-only
THC-RR residual with a reduced-scaling implementation.  The primary source is
Hohenstein *et al.*, *J. Chem. Phys.* **156**, 054102 (2022),
[arXiv:2111.11473](https://arxiv.org/abs/2111.11473).  Equation and algorithm
numbers below refer to that paper.

## Current safe endpoint

The current production-disabled endpoint evaluates

```text
R_candidate = (R_RR - R_123[U_RR]) + R_123[y,T]
```

and shares each T1-transformed Cholesky block between the exact-RR and THC
evaluations of Algorithms 1--3.  This identity is the correctness oracle for
all later work.  It is not a performance implementation: all six exact RR
component kernels and the exact `R_123` subtraction are still evaluated.

Algorithms 1--3 cut across the conventional `Woooo`, `Wvvvv`, and ring
groups.  No whole coarse group may be removed.  A direct replacement is
accepted only after its individual paper equation agrees with the current
oracle.

## Paper boundary

Amplitude THC plus Cholesky-decomposed ERIs covers Algorithms 1--3.  The
paper's direct complement uses:

| Paper path | Contribution | First implementation | Promotion condition |
|---|---|---|---|
| Algorithms 4--5 | Eq. 32 Omega-A and Eq. 33 Omega-C quadratic doubles terms | Two-index-slice O(N^5) GEMM reference | Each raw projected contribution agrees with a dense Eq. 32/33 reference |
| Algorithm 6 | Joint O(N^4) form of Eqs. 32--33 | Custom CUDA only after Algorithms 4--5 pass | Agrees with the O(N^5) implementation and wins end-to-end on WATER4 |
| Algorithm 7 | Omega-D contribution and explicit spurious-term removal | Blocked two-index implementation | Term-level agreement with the safe complement oracle |
| Algorithms 8--10 | Singles and remaining Omega-E/G/H/I/J terms | Matrix contractions in paper order | Singles and doubles residual terms agree separately |

Algorithms 4--7 require only the `ovov` class of ERIs in the paper's THC form,

```text
(ia|jb) = sum(I,J) x_occ[i,I] x_vir[a,I] Z[I,J]
                     x_occ[j,J] x_vir[b,J].
```

They do not require THC factorizations of the `oooo` or `vvvv` classes.

## New interfaces

The first correctness layer is independent of the iterative engine:

```python
ERITHCFactors(x_occ, x_vir, core)

thc_omega_a_algorithm4(y_occ, y_vir, amplitude_core, eri_factors)
thc_omega_c_algorithm5(y_occ, y_vir, amplitude_core, eri_factors)
```

Both kernels return an `(amplitude_thc_rank, amplitude_thc_rank)` raw
projected residual contribution.  They are audit-only until all coefficient,
permutation, and composition checks pass.  They must preserve the input array
backend and may not perform an implicit device-to-host transfer.

The next preprocessing layer will fit the three-index `L[A,i,a]` factors as

```text
L[A,i,a] ~= sum(I) xi[A,I] x_occ[i,I] x_vir[a,I]
Z[I,J] = sum(A) xi[A,I] xi[A,J].
```

The orbital weights must be derived from the MP2 natural-orbital occupation
changes used by [orbital-weighted LS-THC](https://doi.org/10.1063/1.4876016):
the underlying prescription is
`sqrt(abs(MP2 occupation - SCF occupation))`. Any numerical floor is an
explicit approximation control and must be recorded; unit weights are allowed
only for algebraic tests.

The fit tolerance, realized ERI-THC rank, weighted and unweighted residuals,
iteration count, seed, transfer counts, fit time, and storage must be recorded.
The analytic full-pair construction is retained as an exact small-system
endpoint; it is never presented as a scalable WATER8 configuration.

## Acceptance ladder

1. Random real FP64 tensors: factorized Algorithms 4 and 5 agree separately
   with direct Eq. 32 and Eq. 33 contractions.
2. Analytic full-pair amplitude and ERI factors reproduce the corresponding
   RR/CD projected terms on WATER2.
3. Direct Algorithms 4--10 recombine with Algorithms 1--3 to reproduce the
   complete RR residual before any inexact fit is enabled.
4. Inexact ERI and amplitude factors pass WATER2, then WATER4, for projected
   residual, reconstructed full-space residual, energy, and symmetry.
5. Only WATER4 candidates with a complete-iteration speedup may run on
   WATER8.  Factor construction is included in post-HF wall time.

The existing `complete RR - exact R_123` path stays available as an independent
oracle after the direct complement becomes the default experimental path.
