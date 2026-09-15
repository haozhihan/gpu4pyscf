# THC-RR direct complement implementation contract

This note fixes the algebraic boundary for replacing the validation-only
THC-RR residual with a reduced-scaling implementation.  The primary source is
Hohenstein *et al.*, *J. Chem. Phys.* **156**, 054102 (2022),
[DOI 10.1063/5.0077770](https://doi.org/10.1063/5.0077770).
[arXiv:2111.11473v1](https://arxiv.org/abs/2111.11473) is retained as a
versioned preprint, not as the final equation authority. Equation numbers
below state explicitly whether they refer to the preprint or the published
article because the final article both renumbered and corrected the Omega-D
equation.

This ledger includes the published-equation audit at commit `1a9eeb3`. It
distinguishes the existing
Algorithms 1--3 hybrid, the independently audited Algorithms 4--10/F-hat
components, and the from-zero complete audit assembler. The exact full-pair
total residual now agrees with the independent RR/CD oracle after one physical
pair symmetrization of Algorithm 7. A later primary-source audit found that
the final JCP article corrected the disputed arXiv-v1 Eq. 35. The corrected
published Eq. 36 identity now passes one-dimensional, nontrivial physical-pair,
independent scalar-loop, and exact full-pair regressions. No direct-paper
Algorithms 1--10 path is enabled in the iterative
production driver, and every complete, accepted, formal, production, and
performance flag remains false.

## Current safe endpoint

The current production-disabled endpoint evaluates

```text
R_candidate = (R_RR - R_123[U_RR]) + R_123[y,T]
```

and shares each T1-transformed Cholesky block between the exact-RR and THC
evaluations of Algorithms 1--3.  This identity is the correctness oracle for
all later work.  It is not a performance implementation: all six exact RR
component kernels and the exact `R_123` subtraction are still evaluated.
Its legacy `complete_equation_graph=true` metadata is scoped only to the
current RR coarse residual graph by commit `85a3592`; it does not assert a
complete production implementation of paper Algorithms 1--10.

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
| Algorithm 6 | Joint O(N^4) form of Eqs. 32--33 | Python/CuPy GEMM audit endpoint; custom CUDA remains a later performance step | Agrees with the O(N^5) implementation and wins end-to-end on WATER4 |
| Algorithm 7 | Omega-D contribution and explicit spurious-term removal | Audit endpoint for published Eq. 36, Eq. 37, Eq. 38, and Appendix line 19; arXiv-v1 Eq. 35 is a legacy diagnostic | Published Eq. 36 agrees with Eq. 38 plus line 19 in the physical full-pair gauge; later molecular and production gates remain separate |
| Algorithms 8--10 | Singles and remaining Omega-E/G/H/I/J terms | NumPy/CuPy audit contractions in paper order | Independent F-hat construction and complete singles/doubles residuals agree separately |

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
thc_omega_ac_algorithm6(y_occ, y_vir, amplitude_core, eri_factors)
thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri_factors)

build_weighted_eri_thc_preprocess(
    mp2_doubles, amplitude_factors, lov, *, eri_thc_rank,
    fit_tolerance, occupation_floor, ...
)

thc_omega_gh_algorithm8(y_occ, y_vir, amplitude_core, transformed_cholesky)
thc_omega_e_algorithm9(
    y_occ, y_vir, amplitude_core, fhat_oo, fhat_vv, algorithm8
)
thc_omega_ij_algorithm10(
    y_occ, y_vir, amplitude_core, fhat_ov, fhat_vo
)

build_t1_transformed_fhat(
    hcore_mo, integral_provider, t1, *, provenance, ...
)

thc_omega_cd_joint_audit(
    y_occ, y_vir, amplitude_core, eri_factors, ...
)

assemble_complete_thc_ccsd_audit(
    y_occ, y_vir, amplitude_core, tau,
    identity_attested_eri_factors, fhat,
    *, omega_ac_path, orbital_identity_token,
    integral_identity_token, hcore_identity_token, ...
)
```

Algorithms 4--6 return an `(amplitude_thc_rank, amplitude_thc_rank)` raw
projected residual contribution. With final-publication numbering, Algorithm 7
returns its Eq. 37 `R`, exchange `S`, Eq. 38 main term, signed
spurious-removal term, and Appendix line-19 sum separately. The dense joint
audit also evaluates literal published Eq. 36 and the old arXiv-v1 Eq. 35 as
distinct objects. They are audit-only until all coefficient,
permutation, and composition checks pass.  They must preserve the input array
backend and may not perform an implicit device-to-host transfer.

Algorithm 6 currently reproduces both dense preprint Eq. 34 and the sum of the
independent Algorithms 4 and 5 implementations. It is a Python/CuPy GEMM
schedule and carries no CUDA-kernel performance claim. Algorithm 7 reproduces
its projected-amplitude intermediate, main ERI contraction, and explicit
signed spurious-removal contraction. The old literal comparison used arXiv v1
Eq. 35. That preprint equation is unequal to preprint Algorithm 7 line 22,
including in a one-dimensional counterexample. The final JCP article corrects
the disputed factor, as recorded in the primary-source audit below. Commit
`7399a87` independently established that the complete RR/CD doubles residual
requires the raw Algorithm 7 value plus its physical pair transpose. The
published-equation correction now explains that algebraic result and passes
the explicit small-tensor regression gate. Inexact-factor, WATER2/WATER4,
iterative-convergence, accuracy, and performance gates must still pass before
any production path is enabled.

## Published-equation correction and Ref. 107 audit

The author-hosted source for Ref. 107 is Koch *et al.*, *Chem. Phys. Lett.*
**228**, 233--238 (1994),
[A direct atomic orbital driven implementation of CCSD](https://trygvehelgaker.no/Publications/ChemPhysLett_228_233_1994.pdf).
Its printed pages 235--236 give the complete doubles residual in Eq. 32,
Omega-C in Eq. 35, Omega-D in Eq. 36, and the transformed inactive Fock and
two-electron combinations in Eqs. 39--40. Eqs. 26--27 state the relevant
symmetry boundary: the T1-transformed integrals retain bra--ket pair symmetry,
while the usual within-pair/8-fold symmetry is unavailable.

The final JCP article changes the disputed preprint Eq. 35 into published
Eq. 36. In the final equation, the second doubles factor is

```text
2 t[j,l,b,d] - t[j,l,d,b]
```

rather than the preprint's

```text
t[j,l,b,d] - t[j,l,d,b].
```

The final article also renumbers preprint Eqs. 36--37 as published Eqs. 37--38
and compacts Algorithm 7 line 22 into published line 19. Define

```text
A[i,j,a,b] = 2 t[i,j,a,b] - t[i,j,b,a]
S[i,j,a,b] = -t[i,j,b,a]
L[i,a,j,b] = 2 (ia|jb) - (ib|ja).
```

For a physical doubles tensor, `t[i,j,a,b] = t[j,i,b,a]`, and for the
T1-transformed ERIs the allowed bra--ket symmetry is
`(ia|jb) = (jb|ia)`. Under exactly those symmetries, published Eq. 36
projects

```text
1/4 * (A L A + S K S),     K[i,a,j,b] = (ib|ja).
```

Published Eq. 38 is the `1/4 * A L A` term, while Algorithm 7 line 19 adds
`1/4 * S K S`. Thus the final published equation and the algorithm have the
same algebra in the physical full-pair gauge. The earlier `0.25` one-index
difference diagnoses the arXiv-v1 typo: after the correction, both sides are
`0.5` for the all-ones endpoint.

This equivalence is not asserted for arbitrary nonsymmetric test tensors,
because the index relabeling itself uses the physical pair and bra--ket
symmetries. Nonsymmetric fixtures remain useful for checking each contraction
endpoint and must report the equivalence as out of scope. Promotion requires
one-dimensional, nontrivial pair-symmetric, scalar-loop, and exact full-pair
tests against the final published equation. Passing this algebra gate only
removes the equation-version blocker; it does not satisfy inexact-fit,
WATER2/WATER4, convergence, accuracy, or performance gates.

The preprocessing layer fits the three-index `L[A,i,a]` factors as

```text
L[A,i,a] ~= sum(I) xi[A,I] x_occ[i,I] x_vir[a,I]
Z[I,J] = sum(A) xi[A,I] xi[A,J].
```

The audit implementation derives orbital weights from the MP2 natural-orbital
occupation changes used by
[orbital-weighted LS-THC](https://doi.org/10.1063/1.4876016):
the underlying prescription is
`sqrt(abs(MP2 occupation - SCF occupation))`. Any numerical floor is an
explicit approximation control and must be recorded; unit weights are allowed
only for algebraic tests.

`build_factorized_mp2_natural_occupation_weights(...)` evaluates the occupied
and virtual MP2 one-particle-density corrections from an amplitude-THC
reconstruction of RR MP2 doubles without constructing dense `t2`. It records
the RR projector and amplitude-fit identities, the approximation source, the
occupation floor, trace conservation, and all GPU scalar reads. These are
audit weights; they are not represented as the exact full-MP2 density.

`fit_weighted_eri_thc(...)` implements the weighted CP/ALS normal equations.
The fit tolerance, realized ERI-THC rank, weighted and unweighted residuals,
iteration count, seed, transfer counts, fit time, and storage must be recorded.
The analytic full-pair construction is retained as an exact small-system
endpoint; it is never presented as a scalable WATER8 configuration.

`build_weighted_eri_thc_preprocess(...)` binds the RR MP2 doubles, amplitude
factors, orbital weights, natural-orbital rotations, and `L[A,i,a]` object by
identity. It rotates `L` into the MP2 natural-orbital basis, performs the
weighted fit, and rotates the ERI factors back to the working MO basis. Its
explicit full-pair endpoint reconstructs the original working-basis `L` and
`ovov` after nontrivial rotations. It remains audit-only and never constructs
dense `t2`, a pair matrix, or a four-index ERI.

Algorithms 8--10 expose Omega-G/H singles, the Omega-E doubles core, and
Omega-I/J singles separately. Published Eq. 39 requires the occupied orientation
`Lhat_ji D_ja`, whereas Appendix Algorithm 8 line 13 prints `Lhat_ij D_ja`.
Both are retained, but only the published Eq. 39 result is the current
equation-facing output. Algorithm 10's formal published Eq. 43 equivalence
requires a symmetric amplitude core; nonsymmetric cores exercise only the
literal appendix schedule. The final JCP version of record numbers Algorithm
8 as Eqs. 39--41, Algorithm 9 as Eqs. 40--42, and Algorithm 10 as Eq. 43; the
corresponding arXiv-v1 numbers 38--40, 39--41, and 42 remain legacy metadata.
The complete Algorithms 1--10 ledger records equations that each endpoint
directly emits: Algorithm 8 has published Eqs. 39--41, Algorithm 9 has Eq. 42,
and Algorithm 10 has Eq. 43. Algorithm 9's endpoint metadata additionally
lists Eqs. 40--41 because it consumes those Algorithm 8 intermediates; the
ledger does not repeat consumed dependencies.

`build_t1_transformed_fhat(...)` (commit `a1adb89`) now provides the separate
inactive F-hat audit required by Algorithms 8--10. It applies the T1 orbital transforms to
the one-electron matrix and three-index factors, forms the occupied Coulomb
and exchange contribution in auxiliary blocks, and does not materialize a
four-index ERI or dense doubles. It accepts only explicitly symmetric
three-index provider output by default. Any raw-factor symmetry tolerance is
an explicit empirical absolute tolerance, after which the accepted factors
are symmetrized. Orbital, integral, and one-electron identities are bound by
strict JSON-safe provenance, while host-to-device coefficient uploads require
an explicit transfer counter. This builder is independently audited and is
not yet wired into a production direct residual.

## Complete residual assembly boundary

The direct paper route must start from zero. Its doubles core is

```text
S2_thc = Algorithm1 + Algorithm2 + Algorithm3
       + ((Algorithm4 + Algorithm5) XOR Algorithm6)
       + (Algorithm7 + pair_transpose(Algorithm7)) + Algorithm9
S2_rr = tau @ S2_thc @ tau.T
```

The back-projection occurs exactly once. Algorithms 4 plus 5 and Algorithm 6
are mutually exclusive implementations of the same Omega-A/Omega-C terms.
Singles are `Algorithm8.singles_gh + Algorithm10.singles_ij` and are not
back-projected. Every algorithm-level addend has external coefficient `+1`;
Eqs. 28--42 already contain the required signs, factors, and permutations.

The existing hybrid `complete RR - exact R_123 + THC R_123` remains an
oracle. Algorithms 4--10 cannot be appended to that hybrid because its six RR
coarse groups already contain those diagrams. The paper outputs equation
residuals, while the current RR builders expose numerators; comparisons must
subtract or add the projected orbital-energy action before asserting
equivalence.

`assemble_complete_thc_ccsd_audit(...)` (commits `5987560` and `7399a87`) now
makes this boundary executable as an audit graph.
It starts from a zero THC doubles core, enforces the Algorithms 4+5 versus
Algorithm 6 XOR, calls the shared RR back-projection exactly once, and keeps
the singles outside that projection. The raw Algorithm 7 result and its pair
transpose are added once before that back-projection. It requires matching
caller-attested orbital, integral, and one-electron identity tokens across the
ERI factors and F-hat context. Those tokens do not prove the numerical arrays'
provenance or make them immutable, which the metadata states explicitly. The
deprecated `require_eq35_equivalence` name is retained only as a physical
full-pair gauge precondition and version-of-record contract assertion; the
independent dense joint audit establishes the actual published Eq. 36 numerical
identity. Every complete, accepted, formal, production, and performance flag
remains hard-coded false because no inexact-factor or molecular production gate
has passed.

## Current audit status

- Algorithms 4 and 5 pass independent dense Eq. 32 and Eq. 33 references.
- Algorithm 6 passes dense Eq. 34 and exactly matches Algorithms 4 plus 5 for
  nonsymmetric amplitude and ERI cores.
- Algorithm 7 passes published Eq. 37, Eq. 38, and the signed Appendix line-19
  removal. The independent dense joint audit proves that their sum equals
  literal published Eq. 36 in the physical full-pair gauge, including a
  one-dimensional endpoint, a nontrivial pair-symmetric case, and a scalar-loop
  oracle. Commit `4b09acd` remains valid historical evidence that arXiv-v1
  Eq. 35 is unequal, including the all-ones difference of `0.25`; that old
  expression is now an explicit legacy diagnostic. Commit `7399a87` adds the raw
  Algorithm 7 contribution plus its physical pair transpose exactly once. Six
  exact full-pair cases, including nonzero T1 and both Algorithms 4+5 and
  Algorithm 6 paths, reduce the maximum doubles total-identity error from
  `7.746556e-3` before the fix to `1.38778e-16`; the maximum singles error is
  `5.55112e-17`, and Algorithm 6 minus Algorithms 4+5 is at most
  `3.46945e-18`.
- Commit `1a9eeb3` passes 76 focused CPU tests with seven GPU-environment skips.
  Commit `bfe9fc8` then adds four compressed amplitude-THC equation-identity
  cases with `RR rank < OV`, `THC rank < OV`, dense non-one-hot CP factors,
  nonzero T1, and both complete A/C paths.  Against an independently
  reconstructed RR/CD oracle, the maximum doubles and singles differences are
  `1.20e-15` and `1.11e-16`; the complete local THC family now passes 260 tests
  with 34 GPU-environment skips.
  An independent re-review closed three documentation and gate-semantics issues
  and found no remaining P0--P2 defect. These counts establish local algebra
  behavior only; A100 validation of this commit is still missing.
- Algorithms 8--10 pass independent dense published Eq. 39--43 audits, with
  the published Eq. 39 versus Appendix line-13 and symmetric-core boundaries
  recorded explicitly. The arXiv-v1 Eq. 38--42 numbering is retained only as
  versioned legacy metadata.
- The provenance-bound T1-transformed inactive F-hat builder passes independent
  dense and provider audits without constructing four-index ERIs or doubles.
- The Algorithms 1--10 audit assembler enforces the from-zero ledger, XOR
  branch, one physical pair symmetrization of Algorithm 7, one doubles
  back-projection, unprojected singles, shared transfer counter, and
  cross-artifact identity tokens. Its exact full-pair total-identity fixture
  passes. The equation-version contract is now resolved, but complete,
  accepted, formal, production, and performance flags remain false.  The new
  compressed-amplitude identity is defined in the projector reconstructed from
  that CP fit; the fit itself deliberately fails its accuracy gate, the ERI
  side remains analytic full-pair/CD, and WATER2/WATER4 molecular gates have
  not passed.
  Caller-attested identity tokens neither prove the numerical arrays' origin
  nor make those arrays immutable, and the assembler reports both limitations.
- Weighted ERI-THC CP/ALS, an exact-pair endpoint, and factorized MP2
  occupation weights are connected by an audit-only preprocessing seam.
- MTU job 62651 passed 140 A100 CUDA audit tests with six environment skips;
  it ran on a non-target development node and provides no performance claim.
  Its source predates commits `a1adb89`, `4b09acd`, and `5987560`, so it does
  not qualify the F-hat, joint Omega-C/Omega-D, or complete-assembler additions.
- None of these components is connected to the production iterative driver,
  and none supports a WATER8 speed claim.

## Acceptance ladder

1. Random real FP64 tensors: factorized Algorithms 4 and 5 agree separately
   with direct Eq. 32 and Eq. 33 contractions.
2. Analytic full-pair amplitude and ERI factors reproduce the corresponding
   RR/CD projected terms on WATER2.
3. Direct Algorithms 1--10 assemble from zero, use one doubles
   back-projection, and reproduce the complete RR residual before any inexact
   fit is enabled. The full-rank gate includes a nontrivial exact THC gauge,
   not only pair one-hot factors.
4. Inexact ERI and amplitude factors pass WATER2, then WATER4, for projected
   residual, reconstructed full-space residual, energy, and symmetry.
5. Only WATER4 candidates with a complete-iteration speedup may run on
   WATER8.  Factor construction is included in post-HF wall time.

The existing `complete RR - exact R_123` path stays available as an independent
oracle after the direct complement becomes the default experimental path.
