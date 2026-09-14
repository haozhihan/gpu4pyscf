# A100 WATER27 CCSD harness

## Rebuilding the bounded GINT SM80 runtime

`build_gint_sm80_bundle.py` builds the bounded selected-GINT library from a
normalized, identity-bound MTU build input and assembles a content-addressed
13-library runtime bundle. It requires the source SHA-256 and digestible-file
count returned by `stage_mtu_build_input.sh`, plus the pinned 13-library base
bundle. The default mode is a fail-closed plan and performs no build or write:

```bash
python build_gint_sm80_bundle.py \
  --source-root /path/to/build-inputs/STAGING/source \
  --source-sha256 SHA256 \
  --source-file-count COUNT \
  --base-bundle /path/to/base-runtime-bundle \
  --output-task-root /path/to/task
```

Add `--execute` only after reviewing the plan. The plan records the pinned
tool paths `/usr/local/cuda-12.8/bin/nvcc` and
`/usr/local/cuda-12.8/bin/cuobjdump`; execute mode checks those tools before
starting CMake and rejects either tool unless its version banner is CUDA 12.8.
The tool configures CUDA 12.8 with `RelWithDebInfo` and
`80-real`, then verifies the seven exported GINT ABI symbols, the v2
size/offset/workspace contract, sm80-only cubins, dependency resolution, and
selected Rys-7/8 stack frames (at most 4096 bytes). ABI values are probed in a
short-lived Python subprocess so the temporary build library is unmapped before
cleanup. It writes
an immutable `build-verification.json` and `build.log` under
`builds/<source-sha>-gint-bounded-workspace-v2-sm80`. The runtime bundle is
published as exactly 13 `.so` files under `runtime-bundles/<bundle-id>`; its
independent sidecar is `runtime-bundle-manifests/<bundle-id>.json`. The bundle
ID is the established SHA-256 of the canonical JSON inventory list with no
trailing newline. Existing bundle, sidecar, or build paths are never
overwritten, and a failed multi-artifact publish is rolled back.

The candidate workflow is ordered as stage, build, then deploy. On the local
machine, freeze the exact worktree and retain its JSON receipt:

```bash
CCSD_TASK_PATH=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
STAGE_JSON="$(MTU_TASK_ROOT="${CCSD_TASK_PATH}" ./stage_mtu_build_input.sh)"
REMOTE_SOURCE_ROOT="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["remote_source_root"])' \
  <<<"${STAGE_JSON}")"
SOURCE_SHA256="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["source_sha256"])' \
  <<<"${STAGE_JSON}")"
SOURCE_FILE_COUNT="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["source_file_count"])' \
  <<<"${STAGE_JSON}")"
```

In an MTU shell, use those three values to build against the pinned base
bundle. Capture the builder's JSON output; its `bundle_path` is the only valid
candidate deployment input:

```bash
CCSD_TASK_PATH=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
MTU_PYTHON_BIN=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python
REMOTE_SOURCE_ROOT=/path/from-STAGE_JSON/remote_source_root
SOURCE_SHA256=SHA256_FROM_STAGE_JSON
SOURCE_FILE_COUNT=COUNT_FROM_STAGE_JSON
BUILD_JSON="$("${MTU_PYTHON_BIN}" \
  "${REMOTE_SOURCE_ROOT}/benchmarks/cc/a100_water8/build_gint_sm80_bundle.py" \
  --source-root "${REMOTE_SOURCE_ROOT}" \
  --source-sha256 "${SOURCE_SHA256}" \
  --source-file-count "${SOURCE_FILE_COUNT}" \
  --base-bundle "${CCSD_TASK_PATH}/runtime-bundles/BASE_BUNDLE_ID" \
  --output-task-root "${CCSD_TASK_PATH}" \
  --execute)"
RUNTIME_BUNDLE_ROOT="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["bundle_path"])' \
  <<<"${BUILD_JSON}")"
```

Back on the local machine, pass that exact absolute MTU path explicitly. The
candidate profile has no fallback to the baseline virtual environment:

```bash
RUNTIME_BUNDLE_ROOT=/path/from-BUILD_JSON/bundle_path
DEPLOY_JSON="$(MTU_TASK_ROOT="${CCSD_TASK_PATH}" \
  MTU_RUNTIME_BUNDLE_ROOT="${RUNTIME_BUNDLE_ROOT}" \
  DEPLOYMENT_PROFILE=candidate \
  ./deploy_mtu_snapshot.sh)"
CCSD_SNAPSHOT_PATH="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["source"])' \
  <<<"${DEPLOY_JSON}")"
```

Deployment opens the content-addressed bundle, its
`runtime-bundle-manifests/<bundle-id>.json` sidecar, and
`builds/<source-sha>-gint-bounded-workspace-v2-sm80/build-verification.json`
through descriptor-anchored, no-follow reads. It verifies the exact 13-library
inventory and source identity before copying a library. The sealed snapshot
records the bundle ID plus the sidecar, detached build-verification, inventory,
and compact validation-evidence hashes. The `g0-canonical-pristine` profile is
the sole legacy exception and uses the explicit
`MTU_GPU4PYSCF_BINARY_ROOT`; it does not claim candidate bundle validation.

This directory contains a reproducible harness for water2 and water4 validation,
the `H2O8d2d` performance target, and the independent `H2O8s4` accuracy
holdout. The first three geometries are carried over from the prior MTU case
file. The holdout is copied from the official
[GMTKN55 repository](https://github.com/grimme-lab/GMTKN55) at the revision and
file SHA-256 recorded in `cases.json`. Coordinates are in angstroms and all
cases use spherical `cc-pVTZ`.

The driver supports seven explicit method labels:

* `canonical`: GPU4PySCF canonical CCSD.
* `canonical_legacy`: the transfer-heavy canonical execution path used only
  for the shared-orbital G1 A/B comparison.
* `fno`: MP2 frozen-natural-orbital CCSD with the recorded Delta-MP2 value.
* `rr_cd`: compressed RRCCSD driven by the direct Cholesky provider.
* `thc_cd`: the direct-Cholesky R123/complement full-pair amplitude-THC
  validation endpoint. The production proof path shares each T1-transformed
  Cholesky block between exact-RR and THC R123 contractions and constructs the
  complement without calling the complete-RR wrapper. It remains
  performance-ineligible because all six exact RR component kernels and the
  exact R123 subtraction oracle are still evaluated, and inexact THC is
  fail-closed.
* `rr_canonical`: RRCCSD using the dense canonical residual for validation.
* `thc_canonical`: the dense two-level THC validation surrogate.

The paper-equation boundary and staged Algorithms 4--10 acceptance rules are
recorded in [thc-direct-complement.md](thc-direct-complement.md).

The THC audit implementation is current through commit `7399a87`. Commit
`a1adb89` supplies the provenance-bound T1-transformed F-hat builder; `4b09acd`
runs the joint Omega-C/Omega-D audit and records that the current Eq. 35 versus
Eq. 37/Appendix-line-22 conventions remain unequal even for physical
pair-symmetric gauges; `85a3592` confines legacy complete-graph metadata to the
current RR coarse graph; and `5987560` assembles Algorithms 1--10 from zero with
one doubles back-projection. Commit `7399a87` adds the missing physical pair
transpose to Algorithm 7 during complete assembly. Across six exact full-pair
FP64 cases, including nonzero T1 and both Algorithms 4+5 and Algorithm 6 paths,
this reduces the maximum doubles total-identity error from `7.746556e-3` to
`1.38778e-16`; the maximum singles error is `5.55112e-17`, and the maximum
Algorithm 6 minus Algorithms 4+5 error is `3.46945e-18`. This establishes the
assembled full-pair total identity for those fixtures, but it does not establish
the literal Eq. 35 mapping. The assembler remains an audit endpoint, and all
direct-paper complete, accepted, formal, production, and performance flags are
false.

Every reduced-rank or THC record states its validation status and remains
performance-ineligible until the corresponding numerical, allocation, and
A100 timing gates pass.  `--rr-ring-kernel reference|gemm` selects the RR ring
validation kernel for RR methods. Job 62676, from the earlier `77a1b7c` source,
supplies a one-cycle WATER4 reference/GEMM A/B. Its ring times are within 5%,
so the declared tie rule selects the GEMM/cuBLAS path if a revised RR route
returns to a converged backend comparison. Commit `64d6c55` then makes that
GEMM implementation choose residual-first or right-project-first matrix chains
independently for each actual full or tail tile, allowing the latter only when
it uses fewer scalar multiplies without exceeding the residual-first logical
temporary peak; the fourth Wvovo contraction retains its separate bounded
chain. Job 62676 predates this adaptive selector, which still needs an A100
numerical/allocation check.

Commit `30ddc82` additionally fuses the two projected Wvvvv ladders through the
exact symmetric bilinear identity
`sym(X @ sym(core) @ (X - 2Y).T)`. It caps each fused auxiliary subblock at
half the actual outer-block width so the two simultaneously live rank-square
transforms remain within the previous one-transform allowance; a singleton
tail uses the sequential path. Exact algebra and local tests pass, including
the workspace ledger. The fused kernel still needs A100 numerical, allocation,
and performance gates, and it cannot repair job 62674's energy failures. These
kernel choices remain performance-ineligible. THC uses a separate residual
engine and rejects `--rr-ring-kernel gemm` rather than silently ignoring it. No
speedup claim is made by this harness.

A clean local release worktree at `release-30ddc82a33ea` passed 1023 tests
with 74 environment skips. The upstream `gpu4pyscf/cc/tests/test_ccsd.py` was
excluded because local PySCF is unavailable. This validates the local source
state; it does not replace the pending A100 gates.

Use `--dry-run` to inspect the full method/config plan without importing GPU
libraries or running SCF:

```bash
python benchmark.py --case water2-tz --method canonical --device gpu --dry-run
python make_threshold_grid.py --dry-run
python self_check.py
```

Normal runs write one JSON result and one log per case/method/repeat and refuse
to overwrite either file.  The post-HF boundary begins after converged HF and
includes integral transformation, FNO/RR/THC setup, CC iterations, final
residual checks, and one normal checkpoint. GPU streams synchronize at phase
boundaries. RunMetrics phase data, process HBM
sampling, host RSS, CPU affinity, Slurm metadata, CUDA topology, tolerances,
geometry hash, energies, amplitude norms, and available projection residuals
are recorded. A Git worktree records both its commit and content-tree SHA-256;
the normalized regular-tree deployment records the same deterministic content
manifest plus the frozen base commit, so a remote timing never has an unknown
source version.

For RR/THC, the normal timed residual is the labelled projected equation
residual in the RR pair subspace. Final acceptance also requires one reconstructed
full-active-pair equation-residual diagnostic. That diagnostic runs after the
normal post-HF timer and records both `included_in_normal_timed_path: false` and
`included_in_post_hf: false`; its own `wall_seconds` is reported separately and
does not change `post_hf_seconds`. On MTU, set
`RUN_FULL_SPACE_DIAGNOSTIC=1` to pass `--run-full-space-diagnostic` for the
currently supported `rr_cd` and `thc_cd` methods. The launcher accepts only `0`
or `1`. Its default is `0`, which is useful for development timing but cannot
produce an RR/THC record eligible for final approximation acceptance. Legacy
scalar residual fields are displayed by the analyzer but never satisfy either
labelled residual gate.

Canonical legacy/resident A/B validation must use byte-identical RHF orbitals.
The first fresh process writes and immediately reloads a source- and
case-bound artifact; the second fresh process runs its own SCF, validates the
SCF energy, then applies the same serialized orbitals before the post-HF timer:

```bash
python benchmark.py --case water4-tz --method canonical_legacy \
  --device gpu --repeat legacy-a --output-dir results/g1 \
  --orbital-artifact-out results/orbitals/water4-tz.npz
python benchmark.py --case water4-tz --method canonical \
  --device gpu --repeat resident-a --output-dir results/g1 \
  --orbital-artifact-in results/orbitals/water4-tz.npz
python compare_canonical_checkpoints.py \
  results/g1/water4-tz__canonical_legacy__rlegacy-a.json \
  results/g1/water4-tz__canonical__rresident-a.json \
  --output results/g1/checkpoint-parity.json
```

The normal checkpoint is restart-v2 and embeds the exact FP64 MOs, their
fingerprint, and the orbital-artifact SHA-256. The comparator refuses orbital
rotations and compares raw `t1`/`t2` only when the MOs are byte-identical. Its
G1 gates are `1e-8 Eh` for correlation energy, `1e-6` for the final residual
difference, and `1e-6` for amplitude maximum and L2 errors; iteration counts
are diagnostic. The result analyzer separately applies `1e-8 Eh` to both
correlation and total energy for equation-preserving methods.

## G1 shared-orbital canonical A/B gate

`g1_ab.py` fixes the G1 order to `water2 -> water4 -> water8`.  Every case is
one fresh-process `canonical_legacy` producer followed by one dependent
GPU-resident `canonical` consumer.  The producer writes and reloads a
serialized RHF orbital artifact; the consumer loads that exact artifact.  All
six jobs are pinned to `compute-1-6` and form one logical `afterok` chain.

Submission is split at the water4 boundary.  This prevents a successful Slurm
exit from being mistaken for a scientific pass: the first execution can submit
only the four water2/water4 jobs, and water8 remains locked until the analyzer
has validated both pairs and the executed gate receipt.  Generate a plan and
preview all six commands as follows.  `--initial-dependency afterok:62434` is
optional and attaches the first water2 job to an existing Slurm job.

```bash
CCSD_TASK_PATH=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
CCSD_SNAPSHOT_PATH="${CCSD_TASK_PATH}/snapshots/SHA256/source"
CCSD_G1_PATH="${CCSD_TASK_PATH}/results/g1-ab/g1-SHA256"
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/g1_ab.py" plan \
  --task-root "${CCSD_TASK_PATH}" \
  --source-root "${CCSD_SNAPSHOT_PATH}" \
  --run-id g1-SHA256 \
  --initial-dependency afterok:62434 \
  --output "${CCSD_G1_PATH}-plan.json"
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/g1_ab.py" submit \
  --plan "${CCSD_G1_PATH}-plan.json"
```

After reviewing the preview, submit only the small-case gate.  Both plan and
receipt paths are write-once.  The CLI checks the receipt path, including a
dangling symlink, before its first `sbatch` call; an occupied path therefore
causes zero submissions.

```bash
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/g1_ab.py" submit \
  --plan "${CCSD_G1_PATH}-plan.json" \
  --stage gate --execute \
  --receipt "${CCSD_G1_PATH}-gate-receipt.json"
```

Synchronize the complete G1 result directory, including its `timing/` and
`orbitals/` subdirectories, then create the gate analysis.  The relative tree
must remain intact; alternatively run the analysis in place on MTU.  This lets
the analyzer verify the remote output declarations and the synchronized local
JSON, checkpoint, and orbital artifact paths independently.  The executed gate
receipt is mandatory.  Each record's Slurm job ID must equal the concrete ID
stored for that logical job in the receipt.

The comparator requires matching immutable source and hardware
provenance, byte-identical canonical MOs, `|Delta E_corr|` and
`|Delta E_total| <= 1e-8 Eh`, both final residual/update norms and the
legacy/resident residual and amplitude deltas `<= 1e-6`, HBM `<= 72 GiB`, and
host RSS `<= 110 GiB`.  The resident record must show zero
host-staging fallbacks, zero host-staging transfer operations, one resident
iteration per CC iteration, and no H2D, D2H, total-byte, or transfer-count
regression relative to its legacy partner.

```bash
python benchmarks/cc/a100_water8/g1_ab.py analyze \
  --plan /path/to/g1-SHA256-plan.json \
  --gate-receipt /path/to/g1-SHA256-gate-receipt.json \
  --records /path/to/synchronized/g1-SHA256 \
  --output /path/to/g1-SHA256-gate-analysis.json
```

The analysis records the receipt's file hash, canonical payload hash, plan
hash, logical jobs, Slurm IDs, repeats, and expected outputs.  Its portable
binding uses these content identities, so an exact receipt copied with the
result tree remains valid even when its local pathname changes.

Only an analysis with passing water2 and water4 rows can unlock the water8
stage.  Preview by omitting `--execute`; submit by supplying a new receipt.

```bash
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/g1_ab.py" submit \
  --plan "${CCSD_G1_PATH}-plan.json" --stage water8 \
  --gate-analysis "${CCSD_G1_PATH}-gate-analysis.json" \
  --prior-receipt "${CCSD_G1_PATH}-gate-receipt.json"
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/g1_ab.py" submit \
  --plan "${CCSD_G1_PATH}-plan.json" --stage water8 --execute \
  --gate-analysis "${CCSD_G1_PATH}-gate-analysis.json" \
  --prior-receipt "${CCSD_G1_PATH}-gate-receipt.json" \
  --receipt "${CCSD_G1_PATH}-water8-receipt.json"
```

After synchronizing the water8 pair, rerun `analyze` with both executed
receipts and `--require-water8`:

```bash
python benchmarks/cc/a100_water8/g1_ab.py analyze \
  --plan /path/to/g1-SHA256-plan.json \
  --gate-receipt /path/to/g1-SHA256-gate-receipt.json \
  --water8-receipt /path/to/g1-SHA256-water8-receipt.json \
  --records /path/to/synchronized/g1-SHA256 \
  --require-water8 \
  --output /path/to/g1-SHA256-final-analysis.json
```

The water8 submit step also checks that the gate analysis contains the exact
content identity of the `--prior-receipt` supplied at submission time.  The
final G1 A/B result is a single-pair development gate; the final speed claim
still requires three interleaved fresh-process baseline/candidate pairs.

## FNO G2 water4 probe

The staged interface now has three fail-closed phases.  `--stage water2`
creates the complete seven-threshold water2 A/B plus counterpoise grid.
`--stage water4` requires both a passing water2 analysis and its executed
write-once receipt; both files are recorded by path, payload hash, and file
hash in the plan.  `--stage water8` accepts only a complete passing water4
analysis and derives exactly the first passing threshold plus its immediately
tighter neighbor.  Its eight-job dry-run DAG contains two fresh canonical/FNO
timing pairs and two canonical/FNO counterpoise pairs.  Plans are rejected if
either prerequisite is changed after planning, if source digests differ, or if
the FNO active/frozen lists do not form the exact global-MO partition
`[nocc, nocc+nvir)` for the selected WATER27 case.

For example, the progression is:

```bash
python fno_g2.py plan --stage water2 --task-root "$CCSD_TASK_PATH" \
  --source-root "$CCSD_SNAPSHOT_PATH" --run-id g2-water2-SHA256 \
  --output "$CCSD_TASK_PATH/results/fno-g2/water2-plan.json"
python fno_g2.py analyze --stage water2 --records "$CCSD_TASK_PATH/results/fno-g2/water2" \
  --receipt "$CCSD_TASK_PATH/results/fno-g2/water2-receipt.json" \
  --output "$CCSD_TASK_PATH/results/fno-g2/water2-analysis.json"
python fno_g2.py plan --stage water4 --task-root "$CCSD_TASK_PATH" \
  --source-root "$CCSD_SNAPSHOT_PATH" --run-id g2-water4-SHA256 \
  --prior-analysis "$CCSD_TASK_PATH/results/fno-g2/water2-analysis.json" \
  --prior-receipt "$CCSD_TASK_PATH/results/fno-g2/water2-receipt.json" \
  --output "$CCSD_TASK_PATH/results/fno-g2/water4-plan.json"
python fno_g2.py analyze --stage water4 --records "$CCSD_TASK_PATH/results/fno-g2/water4" \
  --receipt "$CCSD_TASK_PATH/results/fno-g2/water4-receipt.json" \
  --output "$CCSD_TASK_PATH/results/fno-g2/water4-analysis.json"
python fno_g2.py plan --stage water8 --task-root "$CCSD_TASK_PATH" \
  --source-root "$CCSD_SNAPSHOT_PATH" --run-id g2-water8-SHA256 \
  --prior-analysis "$CCSD_TASK_PATH/results/fno-g2/water4-analysis.json" \
  --prior-receipt "$CCSD_TASK_PATH/results/fno-g2/water4-receipt.json" \
  --output "$CCSD_TASK_PATH/results/fno-g2/water8-plan.json"
```

The original no-argument `make_plan`/water4 preview remains available for
older local fixtures; formal staged runs should use the explicit stage CLI.

Formal analysis also requires the receipt used to submit the stage.  The
analysis JSON embeds the receipt and plan paths, file hashes, payload hashes,
run ID, and complete logical-to-Slurm mapping.  Counterpoise outputs carry the
same logical job identity plus their Slurm job name and ID; a path-only or
hand-written CP JSON cannot satisfy the gate.  The canonical orbital artifact
and shared CP orbital bundle must exist at the immutable plan path and match
their recorded SHA-256 values.

`fno_g2.py` fixes the FNO occupation-threshold grid to
`{1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7}` and creates an auditable MTU
job graph. Every threshold gets a fresh-process canonical/FNO A/B pair. The
canonical process writes and reloads an orbital artifact; its dependent FNO
process loads that exact artifact. The seven pairs are serialized on one
named node so node-to-node variation cannot be mistaken for an FNO speedup.
The counterpoise chain starts only after the final timing pair, which prevents
a four-GPU node from co-scheduling validation traffic beside an A/B timing
run. One additional canonical counterpoise process writes the complete cluster and
ghost-monomer orbital bundle, which all seven FNO counterpoise processes load.

Generate and inspect a plan only after the candidate snapshot has been sealed:

```bash
CCSD_TASK_PATH=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
CCSD_SNAPSHOT_PATH="${CCSD_TASK_PATH}/snapshots/SHA256/source"
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/fno_g2.py" plan \
  --task-root "${CCSD_TASK_PATH}" \
  --source-root "${CCSD_SNAPSHOT_PATH}" \
  --run-id g2-water4-SHA256 \
  --node compute-1-6 \
  --output "${CCSD_TASK_PATH}/results/fno-g2/g2-water4-SHA256-plan.json"
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/fno_g2.py" submit \
  --plan "${CCSD_TASK_PATH}/results/fno-g2/g2-water4-SHA256-plan.json"
```

The second command prints the exact `sbatch` argv and dependency wiring. To
submit the reviewed plan on the MTU login node, add `--execute` and a new
receipt path:

```bash
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/fno_g2.py" submit \
  --plan "${CCSD_TASK_PATH}/results/fno-g2/g2-water4-SHA256-plan.json" \
  --execute \
  --receipt "${CCSD_TASK_PATH}/results/fno-g2/g2-water4-SHA256-submission.json"
```

After the result directory is synchronized, audit it locally or on MTU:

```bash
python benchmarks/cc/a100_water8/fno_g2.py analyze \
  --records /path/to/results/fno-g2/g2-water4-SHA256 \
  --output /path/to/results/fno-g2/g2-water4-SHA256-analysis.json
```

For each threshold the analysis reports complete post-HF time and speedup,
phase totals, active virtual count, the full/FNO MP2 energies and Delta-MP2,
the Delta-MP2-corrected CCSD correlation error, active-space residual, HBM, and
host RSS. CP-enabled stages separately report the shared-bundle counterpoise
error; the WATER4 stage has no CP calculation. The analysis checks the algebraic
identity `Delta-MP2 = E_MP2(full) - E_MP2(FNO)` and the corresponding corrected
CCSD energies. The water8 candidate list is emitted only when all seven
threshold records are present and both the first loose-to-tight passing point
and its immediately tighter neighbour independently satisfy `1e-4 Eh`,
`0.02 kcal/mol`, residual, resource, exact-orbital, immutable-source, hardware,
and positive water4 speedup gates. This is a development probe; final timing
still uses three interleaved fresh-process baseline/candidate pairs.

The WATER4 gate itself tests correlation energy, equation convergence,
resources, and complete-iteration speed. The formal eight-ghost-monomer
counterpoise calculation is a WATER8 gate and is not part of WATER4. The
historical WATER2 FNO screen below used its separate two-monomer CP probe.

### Current development FNO screening

Jobs 62630--62633 ran a non-formal screening on idle A100 development nodes
from source `b29a65a`. These timings cannot enter the fixed-node performance
claim, but they are sufficient to apply the WATER4 stop rule:

| threshold | WATER4 frozen virtuals | abs. correlation error (Eh) | post-HF (s) | WATER4 decision |
|---:|---:|---:|---:|---|
| `3e-5` | 33 / 212 | `9.3776e-4` | 53.278 | energy fails |
| `1e-5` | 4 / 212 | `5.7596e-5` | 56.355 | accurate but slower than canonical |
| `3e-6` | 0 / 212 | `1.5209e-8` | 58.559 | accurate but no compression and slower |

The earlier, separate WATER2 CP probe measured `0.024753 kcal/mol` at `1e-5`
and `0.001417 kcal/mol` at `3e-6`. It is historical small-case screening and
does not replace the eight-ghost-monomer WATER8 CP gate.

The paired WATER4 canonical observation was 35.433 s. Every tested FNO
candidate was slower, and threshold `3e-6` froze no WATER4 virtual orbitals.
Separately, the WATER2 CP probe rejected `1e-5`. The current FNO route therefore
stops before WATER8; it remains a speed/error probe and is not a candidate for
combination with THC-RR. Formal immutable-source repetition would be required
before changing this conclusion into an acceptance claim.

### Current development RR screening

Jobs 62635, 62637, and 62652 are historical non-formal WATER2 diagnostics from
source `b29a65a`. Job 62637 exposed that the old dense `rr_canonical` update
projected a canonical Jacobi step after dividing by the full-space orbital
denominator, and therefore solved `U^T D^-1 R U = 0` instead of the declared
Galerkin equation `U^T R U = 0`. Commit `5056c8d` replaced it with the projected
Sylvester solve. Job 62653 repeated the complete cutoff grid from immutable
development source `77a1b7c` on an idle A100 and completed with zero record
failures. Its development summary SHA-256 is
`28dfbc5c4db855dfd32ce4d1414726c631deafd40e48a397a5429f9d73f14171`:

| RR cutoff | rank / 1060 | abs. correlation error (Eh) | projected equation residual | reconstructed full-space residual | post-HF (s) |
|---:|---:|---:|---:|---:|---:|
| `1e-3` | 164 | `1.771860e-3` | `9.090670e-8` | `9.764681e-2` | 10.1394 |
| `1e-4` | 276 | `1.614381e-3` | `9.134555e-8` | `7.239172e-2` | 11.0735 |
| `1e-5` | 374 | `1.422687e-3` | `1.419348e-7` | `5.180226e-2` | 11.2621 |
| `1e-6` | 494 | `6.831544e-4` | `1.312658e-7` | `3.348634e-2` | 11.6680 |
| `1e-7` | 604 | `9.002584e-5` | `1.566310e-7` | `1.793478e-2` | 12.5574 |
| `1e-8` | 719 | `1.261283e-5` | `1.270087e-7` | `8.875755e-3` | 13.8979 |
| `1e-9` | 828 | `1.141377e-7` | `6.333480e-7` | `5.856294e-3` | 12.6359 |
| `1e-11` | 965 | `3.086201e-8` | `6.876099e-7` | `1.107442e-3` | 12.8286 |

The first loose-to-tight WATER2 point that satisfies both the `1e-4 Eh`
energy gate and the `1e-6` projected-equation gate is `1e-7`, rank 604; its
immediately tighter neighbour is `1e-8`, rank 719. Both ranks are below the
`0.8 * OV = 848` stop boundary, so they advance to the WATER4 accuracy gate.
The full-space values above are approximation diagnostics and are not used as
the compressed-equation convergence criterion. Job 62637 is retained only as
superseded defect evidence; its energy and projected residual must not be mixed
with the corrected job 62653 grid.

The full-rank `rr_canonical` endpoint in job 62652 remains a separate algebraic
gate. It matches canonical correlation energy within `8.55e-15 Eh`,
reconstructed `t2` within `1.58e-15` max abs., and has equal projected/full
equation residuals of `6.8950e-7`.

The direct-CD Galerkin engine in job 62635 gives the same cutoff-`1e-5` energy
as the corrected canonical-ERI engine within `1.61e-8 Eh`, consistent with its
explicit `1e-8` CD tolerance. Its projected residual is `4.5833e-7`, but the
energy error is `1.422704e-3 Eh`, so that loose candidate remains rejected. Its
WATER2 post-HF time is 1558.248 s: 21.149 s in direct Cholesky construction and
1530.949 s in iterations, including 1351.401 s in doubles and 152.282 s in
singles.

Jobs 62654 and 62656 then ran one-cycle, deliberately non-converged WATER2
profiles from source `77a1b7c`. Both Slurm jobs ended in `FAILED`: the launcher
expected a deliberate `max_cycle=1` non-convergence to return code 5, while the
benchmark actually returned code 1. The result JSON files and CUDA-event
measurements are complete and useful as development observations, but the
scheduler gate did not pass and every record has `performance_eligible=false`.

Job 62654 first varied the virtual block with auxiliary block 1. Its summary
SHA-256 is
`39adefd0937f126ad3221713bd0189c6acce0346a3032967f7617c89cf507a57`:

| ring path | virtual block | post-HF for setup + one cycle (s) | doubles equation (s) | ring component (s) |
|---|---:|---:|---:|---:|
| reference | 8 | 153.733 | 111.505 | 96.406 |
| reference | 16 | 104.928 | 62.759 | 47.928 |
| GEMM | 8 | 155.097 | 112.982 | 97.943 |
| GEMM | 16 | 106.468 | 64.061 | 49.024 |
| GEMM | 32 | 84.466 | 42.409 | 27.437 |

Job 62656 fixed virtual block 32 and swept auxiliary block size. Its summary
SHA-256 is
`73c70837b7ff2dfcdf40cbcaf0b004f0b81925598e6aaa79245f377b2e380a26`:

| ring path | auxiliary block | post-HF for setup + one cycle (s) | doubles equation (s) | ring component (s) | peak process HBM (MiB) |
|---|---:|---:|---:|---:|---:|
| reference | 1 | 84.223 | 42.152 | 27.294 | 69844 |
| reference | 8 | 45.400 | 12.172 | 8.494 | 69846 |
| reference | 16 | 36.616 | 5.882 | 4.109 | 69846 |
| reference | 32 | 32.150 | 2.976 | 2.046 | 70042 |
| GEMM | 1 | 128.414 | 72.352 | 46.740 | 69842 |
| GEMM | 8 | 44.489 | 11.527 | 8.058 | 69844 |
| GEMM | 16 | 36.331 | 5.783 | 3.939 | 69846 |
| GEMM | 32 | 32.161 | 2.951 | 2.045 | 70042 |

At auxiliary block 32 the reference and GEMM observations are effectively
tied, while both are much faster than auxiliary block 1 for this single cycle.
The cross-job GEMM auxiliary-block-1 observations also differ substantially
(84.466 s in job 62654 versus 128.414 s in job 62656), so these data do not
select a production kernel. No value in either table may be extrapolated to a
converged iteration count, WATER4, WATER8, or a 10x whole-stage claim. A fresh
immutable-source, converged run must re-check numerical gates and the 72 GiB HBM
limit before promotion.

Job 62673 performed that next WATER2 check with auxiliary and virtual blocks
both fixed at 32, direct-CD tolerance `1e-8`, and the reference ring path. It
completed all three converged records, wrote normal checkpoints, and passed its
scheduler gate. The summary SHA-256 is
`6c895c36cfc9af248b4b9e0bdfb4957a2dd00b62c4017269800731f03fe7e883`:

| method | cutoff | rank / 1060 | abs. correlation error (Eh) | projected equation residual | full-space diagnostic | iterations | post-HF (s) | peak HBM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | -- | -- | -- | -- | -- | 11 | 8.3873 | 37740 |
| RR-CD | `1e-7` | 604 | `9.0021115e-5` | `6.23137e-7` | `1.793282e-2` | 12 | 63.0650 | 70944 |
| RR-CD | `1e-8` | 719 | `1.2613502e-5` | `6.93585e-7` | `8.874707e-3` | 12 | 87.8201 | 70672 |

Both RR-CD candidates pass the declared WATER2 correlation-energy,
projected-equation, HBM, and host-RSS gates. The full-space residual remains an
approximation diagnostic. Job 62673 ran on development node `compute-1-3` with
`performance_eligible=false`, so its timing selects neither a fixed-node
speedup nor a 10x result. These two candidates advanced to the converged
WATER4 accuracy and iteration-speed gate; job 62674 below records its terminal
result. The eight-monomer ghost-basis
counterpoise validation applies later to WATER8 candidates that pass that gate.

Job 62674 is the resulting direct-CD WATER4 grid from source `77a1b7c` on
development node `compute-1-3`. Both candidate records converged and passed
their projected-equation and resource checks, but both fail the `1e-4 Eh`
WATER4 correlation-energy gate:

| cutoff | rank / 4240 | abs. correlation error (Eh) | projected equation residual | full-space diagnostic | iterations | post-HF (s) | peak HBM (MiB) | peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-7` | 1495 | `5.26532e-4` | `3.9254e-7` | `3.38586e-2` | 14 | 1049.779 | 61242 | 4.291 |
| `1e-8` | 1914 | `1.1056808545e-4` | `4.4503858e-7` | `1.853977e-2` | 14 | 1366.472 | 61238 | 4.271 |

For cutoff `1e-7`, post-HF contains `148.095 s` of direct Cholesky
construction, `23.453 s` of projector construction, and `874.897 s` of
iterations. For cutoff `1e-8`, the corresponding phases are `148.049 s`,
`23.469 s`, and `1190.975 s`; within the 14 iterations, Wvvvv takes
`516.225 s`, Woooo `286.632 s`, the ring component `62.812 s`, and singles
`245.010 s`. Its separate `354.145 s` reconstructed full-space diagnostic is
excluded from post-HF. The paired canonical observation is `35.569 s`; both RR
records are slower. Both include their normal checkpoints. The batch ended
`FAILED 2:0` because its deliberate summary gate counted two scientific
failures, while the process-failure count was zero. The synchronized summary
SHA-256 is
`dba812b465f32f490090cb3a813e413648080535b6441d6839af69d75e913428`.
The WATER4 stop rule therefore ends this RR candidate route, and no RR WATER8
or eight-ghost-monomer counterpoise job will be submitted from these cutoffs.
The failed energy gate already forces that decision, independent of any future
exact Wvvvv kernel fusion.

Job 62676 separately compared the reference and GEMM ring paths for one
deliberately non-converged WATER4 cycle at cutoff `1e-7`, rank 1495, and
auxiliary/virtual blocks 32. Reference versus GEMM post-HF times were
`237.3273 s` and `237.1612 s`; their ring components were `4.4457 s` and
`4.3338 s`. Energy, projected residual, update norm, and amplitude norms agree
to FP64 precision. The difference is below the plan's 5% tie threshold, which
selects the GEMM/cuBLAS path for subsequent testing. The summary SHA-256 is
`36ab6755eca199c4907ef2448d8301fb408e41b4ac093f4a595c586c9a9abadf`.
This is a `performance_eligible=false` component choice from one non-converged
cycle; it cannot be extrapolated to complete iterations or WATER8.

The fixed-node A2 chain did not qualify. Jobs 62612 and 62613 each ran for
seven seconds on `compute-1-6` and ended `FAILED 2:0` with empty logs. Their
short, silent exits are highly consistent with a snapshot-validator rejection
before formal work began, and job 62685 directly reproduces the old
cross-client `st_dev` defect on the same published snapshot. This attribution
for 62612/62613 is a high-confidence inference, because their own logs contain
no direct error text. Jobs 62614 and 62615 then remained pending with
`DependencyNeverSatisfied`; both were explicitly cancelled and superseded at
`2026-09-14T12:27:01` MTU time. The local supersession record has SHA-256
`a643bbf0f3e343a5dec0c9594c84bca8db092889637080767170cd557fc52a1b`.
None of these jobs supplies fixed-node qualification or performance evidence.

Formal timings should run from an immutable deployment made by
`deploy_mtu_snapshot.sh`. It first copies the repository into a private local
staging tree, validates and materializes the allowed relative leaf-file links,
then sends a regular-file-only bundle to one descriptor-anchored remote
receiver. That process verifies the hash, binds the runtime libraries, seals
the tree, and atomically installs a read-only
`snapshots/<sha256>/source` directory. Set
`CCSD_SOURCE_ROOT` to the printed path and `PYTHONDONTWRITEBYTECODE=1` for the
Slurm job. Local filesystems publish with an exclusive no-replace rename. If
the MTU NFS server reports that rename flag as unsupported, the same held
snapshots descriptor creates an exclusive empty directory at the digest name,
binds and rechecks its inode and emptiness, then atomically renames the staging
directory over only that reservation. A pre-existing, replaced, or nonempty
target fails closed; unknown or nonempty reservation content is preserved for
diagnosis. POSIX and NFS provide no inode-conditional rename, so this fallback
uses the explicit trust boundary sealed into the manifest and repeated in its
receipt: processes running as the private-anchor owner UID, and root, are
trusted cooperators. Other users are excluded by the no-link private anchor and
ownership/mode checks. An ambiguous NFS rename error counts as committed only if
the target is the held staging inode and the staging name has disappeared. If
either name or the held descriptor cannot be reconciled, publication is
reported as indeterminate; deployment removes only the identity-bound old
staging name and preserves the digest target while printing recovery
information. A complete publication receipt retains the outcome protocol and
writes a separate, digest-bound `<sha256>.publication-attestation.json` sibling,
leaving the sealed snapshot manifest unchanged. Published digest names are
single-use: deployment refuses an
existing `snapshots/<sha256>` entry and prints its path instead of reopening or
reusing it. Runtime jobs independently require the strict v3 manifest before
using a published snapshot; strict v3 checks the sibling attestation through
descriptor-anchored no-follow reads and binds its protocol, trust boundary,
manifest hash, source digest, and published inode. On a failed fresh deployment, the trap invokes the
no-follow staging guard to remove only the direct child whose device/inode pair
was allocated by that run. After a complete callback receipt, it may also remove
the same inode through the published digest name if later receipt validation
fails. If an ancestor identity changed, cleanup refuses mutation and prints the
retained inode for operator recovery.

Jobs 62678 and 62684 (`rrccsd-gpu-tests-dev`) and diagnostic job 62685 were
rejected on `compute-1-4` before scientific testing because the older validator
required cross-client `st_dev` equality. The publication host attested device
48 and inode 83972272889; `compute-1-4` observed device 53 and the same inode.
Commit `6fe3e9e` keeps the inode as the strict shared-tree binding while treating
the mount-device number as cross-client diagnostic metadata. Validation job
62688 then completed on `compute-1-4` with exit code 0 against the older
published snapshot: `valid=true`, `reasons=[]`, `device_match=false`, and
`inode_match=true`. This verifies the portable validation rule; it is
infrastructure evidence and supplies no CCSD numerical or performance result.

A mutable development overlay is suitable only for correctness and profiling
runs.

The Slurm scripts target account `mri`, partition `mrigpu`, one NVIDIA GPU, and
the measured A100-local NUMA node 3. The compute image has neither `srun` nor
the `numactl` executable, so `topology_guard.py` checks the allocated GPU's
sysfs NUMA node, intersects its CPUs with the Slurm affinity, applies
`sched_setaffinity`, and sets/verifies the kernel `MPOL_PREFERRED` policy with
`libnuma.so.1`. `run_numa3_8core.sbatch` selects physical CPUs `24-31` and
`run_numa3_16thread.sbatch` selects those cores plus SMT siblings `88-95` when
that exact topology was allocated. The guard refuses a performance run if any
check fails and always writes a JSON decision record. Set `PYTHON_BIN` to the
pinned remote environment, set `CASE` and `METHOD`, or pass additional driver
arguments after the script name. `nsys_profile.sh` wraps the same driver with
`/usr/local/cuda/bin/nsys` and refuses to overwrite profiles.

After at least three eligible fresh-process runs in each launch mode,
`compare_topology_runs.py` verifies every benchmark against its independent
topology-guard record, checks scientific and numerical equivalence, computes
the sample CV and median, and applies the frozen 2% tie rule. For example:

```bash
python compare_topology_runs.py \
  --physical8 results/p8-a.json results/p8-b.json results/p8-c.json \
  --logical16 results/l16-a.json results/l16-b.json results/l16-c.json \
  --topology-dir results/topology --output results/topology-comparison.json
```

The audited water2 comparison in `results/mtu/water2-topology-comparison.json`
selected eight physical cores (`24-31`): median 10.810 s versus 11.101 s for
16 logical threads, with sample CVs of 1.50% and 1.72%, respectively. The
water8 acceptance run retains its own topology evidence; water2 is the
development launch-selection gate.

`gpu4pyscf.cc.gint_selected_columns.GINTSelectedAOPairColumnProvider` is the
selected-shell-pair C-ABI candidate. `gint_gate.py` exercises FP64 batches
`B=1,8,32`, the direct packed diagonal, and blocked direct Cholesky with
`B=32` on spherical water2/cc-pVTZ. Selected columns are compared after the
timed boundary with a bounded rectangular `pyscf.ao2mo.general` oracle. The
oracle's largest array has `nao^2 B^2` elements; the gate rejects it before
allocation if it exceeds `--oracle-max-bytes`. The diagonal uses independent
CPU `int2e_sph` shell quartets. Returned GPU shapes, live CuPy pool deltas,
HBM/RSS, C-ABI identity, and operation-level transfers are recorded without
allocating `nao^4`, a square AO-pair matrix, or a per-pivot AO matrix.

The decision separates `correctness_passed`, `qualification_passed`, and
`performance_eligible`. The two-stage contract uses two immutable snapshots.
Qualification snapshot **A** has no release pin. Its first run qualifies CUDA
parity, the complete `_VHFOpt` and timed transfer ledgers, timing boundary, and
resource evidence, but remains performance-ineligible because no release
snapshot is bound to its evidence. Run A through `run_mtu_gint_gate.sbatch`:

```bash
TASK_ROOT="${CCSD_TASK_PATH}"
RESULT_ROOT="${TASK_ROOT}/results/gint-gate"
QUALIFICATION_JOB_ID="$(sbatch --parsable \
  --export="ALL,CCSD_TASK_ROOT=${CCSD_TASK_PATH},CCSD_SOURCE_ROOT=${CCSD_SNAPSHOT_PATH},CCSD_EXPECTED_DEPLOYMENT_PROFILE=candidate" \
  run_mtu_gint_gate.sbatch)"
JOB_ID="${QUALIFICATION_JOB_ID%%;*}"
```

After `sacct` reports that job as `COMPLETED` with exit code `0:0`, issue the
receipt outside the finished allocation. The issuer hashes the terminal Slurm
output, qualification result, topology record, A's source manifest, loaded
`libgint.so`, selected-pair ABI size/offset/version, and both complete transfer
ledgers with `unresolved_count=0`. It creates a new dedicated directory, writes
the receipt and content-hash sidecar with exclusive creation and mode 0444,
then seals that directory as mode 0555. It also writes a deterministic staging
pin. Mode bits are an accidental-write guard; the content-addressed B snapshot
created in the next step is the authoritative receipt anchor.

```bash
python gint_gate_receipt.py \
  --result "${RESULT_ROOT}/water2-gint-direct-cd-${JOB_ID}.json" \
  --slurm-output "${TASK_ROOT}/logs/gint-gate-${JOB_ID}.log" \
  --receipt "${RESULT_ROOT}/gint-runtime-gate-receipt/receipt.json" \
  --release-pin "${RESULT_ROOT}/gint_release_pin.json"
```

Promote A into release snapshot **B**. The promotion copies exactly A,
including the sealed publication policy and its trust-boundary evidence, embeds
the staging pin at `gpu4pyscf/cc/gint_release_pin.json`, recalculates the normal
source digest, writes B's manifest, seals every path, and publishes B with one
atomic rename. B with only that fixed pin omitted must reproduce A's digest.

```bash
python "${CCSD_SNAPSHOT_PATH}/benchmarks/cc/a100_water8/gint_release_snapshot.py" \
  --qualification-source "${CCSD_SNAPSHOT_PATH}" \
  --qualification-manifest "$(dirname "${CCSD_SNAPSHOT_PATH}")/manifest.json" \
  --release-pin "${RESULT_ROOT}/gint_release_pin.json" \
  --task-root "${CCSD_TASK_PATH}" \
  > "${RESULT_ROOT}/gint-release-promotion.json"
CCSD_RELEASE_SOURCE="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["release_source"])' \
  "${RESULT_ROOT}/gint-release-promotion.json")"
```

Run the same gate from B as the release check, passing only the external
receipt path to the provider. The provider rehashes B and B-without-pin, then
rejects mappings, any symlink component, writable snapshot paths, altered
evidence, and any A/B source, manifest, binary, ABI, allocation, or topology
mismatch. The receipt must remain below the same task's `results/` directory:

```bash
RECEIPT_PAYLOAD_SHA="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["payload_sha256"])' \
  "${RESULT_ROOT}/gint-runtime-gate-receipt/receipt.json")"
sbatch --parsable \
  --job-name="gint-release-${RECEIPT_PAYLOAD_SHA}" \
  --export="ALL,CCSD_TASK_ROOT=${CCSD_TASK_PATH},CCSD_SOURCE_ROOT=${CCSD_RELEASE_SOURCE},CCSD_EXPECTED_DEPLOYMENT_PROFILE=candidate,GINT_RUNTIME_GATE_RECEIPT=${RESULT_ROOT}/gint-runtime-gate-receipt/receipt.json" \
  run_mtu_gint_gate.sbatch
```

The pin commits B's source digest to the receipt file hash, payload hash, A
digest and manifest, `libgint.so` hash and size, both ABI identities, the
qualification Slurm job, topology fingerprint, and transfer audit. The full
payload digest in the release job name adds the same identity to the release
allocation record. Replacing both receipt files, or replacing the pin without
publishing a different B digest and manifest, fails closed. Only this second
run from B on the matching qualification topology and current release
allocation can set `performance_eligible=true`.

`gpu4pyscf.cc.gint_pair_columns.GINTAOPairColumnProvider` remains the slower
restricted-task reference adapter for diagnostic comparison.

These scripts only prepare and run jobs. They do not submit remotely.

`water8s4-tz` uses the selected `water8-tz` approximation parameters and must
pass the same energy, counterpoise, residual, HBM, and host-RSS limits. It has
no 10x timing requirement and cannot substitute for the `H2O8d2d` target.

## Counterpoise orbital contract

Formal Boys--Bernardi comparisons use one canonical-orbital bundle containing
the cluster plus every complete-basis ghost monomer.  Each CP process still
runs a fresh RHF calculation.  The first process writes and reapplies its
orbitals before constructing each CC solver; the second process validates and
loads the same FP64 arrays before constructing its solvers:

```bash
python counterpoise.py --case water8-tz --method canonical --device gpu \
  --orbital-bundle-out /external/results/water8-cp-orbitals.npz \
  --output /external/results/water8-cp-canonical.json
python counterpoise.py --case water8-tz --method rr_cd --device gpu \
  --orbital-bundle-in /external/results/water8-cp-orbitals.npz \
  --output /external/results/water8-cp-candidate.json
```

For the MTU Slurm launcher, both jobs must use the same immutable
`CCSD_SOURCE_ROOT`, case, SCF/CC thresholds, and bundle path. Submit the
candidate only after the canonical producer has completed successfully:

```bash
CCSD_TASK_PATH=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
CCSD_SNAPSHOT_PATH="${CCSD_TASK_PATH}/snapshots/SHA256/source"
CCSD_CP_BUNDLE_PATH="${CCSD_TASK_PATH}/results/counterpoise/water8-cp-orbitals.npz"
sbatch --export="ALL,CCSD_TASK_ROOT=${CCSD_TASK_PATH},CCSD_SOURCE_ROOT=${CCSD_SNAPSHOT_PATH},CASE=water8-tz,METHOD=canonical,CP_RUN_ID=cp-oracle,CP_ORBITAL_BUNDLE_OUT=${CCSD_CP_BUNDLE_PATH}" \
  run_mtu_counterpoise.sbatch
# Wait for cp-oracle to finish successfully before submitting the consumer.
sbatch --export="ALL,CCSD_TASK_ROOT=${CCSD_TASK_PATH},CCSD_SOURCE_ROOT=${CCSD_SNAPSHOT_PATH},CASE=water8-tz,METHOD=rr_cd,CP_RUN_ID=cp-candidate,CP_ORBITAL_BUNDLE_IN=${CCSD_CP_BUNDLE_PATH}" \
  run_mtu_counterpoise.sbatch
```

`water8-tz` automatically evaluates one cluster and all eight complete-basis
ghost monomers. `CP_ORBITAL_BUNDLE_IN` and `CP_ORBITAL_BUNDLE_OUT` are mutually
exclusive. The launcher resolves the path, requires it below
`CCSD_TASK_ROOT/results` and outside the source snapshot, requires an input to
exist, and refuses to overwrite an output.

The bundle is bound to the complete atom payload, ghost pattern, active atom
indices, case, basis, RHF charge/spin, dimensions, SCF tolerance, and immutable
source tree. Every entry records its exact MO fingerprint; both CP records
record the bundle SHA-256, bundle fingerprint, entry fingerprints, and a common
`record_match_key`. `shared_bundle_proof()` validates these fields when pairing
the canonical and candidate records. The result analyzer requires every CP
record entering the canonical/candidate medians to pass this proof against one
common bundle; mixed bundles and altered per-entry evidence fail closed. A CP
calculation without an explicit bundle remains a useful diagnostic but records
`accuracy_eligible: false`.
Keep the bundle and result JSON outside the read-only source snapshot.

## Evidence ledger

`implementation-status.json` is the machine-readable gate and MTU job ledger.
It distinguishes `source_reported`, `local_measured`, `derived`, and
`hypothesis` evidence and records why a run is or is not eligible for a timing
claim.  The workspace-level
`output/water8-ccsd-implementation-artifacts.sha256` inventories every raw MTU
artifact that had been synchronized when the ledger was written, plus the
preimplementation evidence files used to freeze the contract.

When a queued or running job finishes, synchronize its result, log, checkpoint,
topology record, and Slurm accounting before updating the ledger.  Add every
new file to the SHA-256 inventory, validate snapshot and topology evidence at
both run boundaries, and only then change its evidence state.  A completed job
does not establish a speedup until the matched repetitions and accuracy gates
also pass.

MTU may assign a 32-CPU cgroup that excludes the GPU-local physical cores even
when the requested GPU is on NUMA node 3.  Current launchers therefore reserve
64 CPUs and let the topology guard narrow actual execution to eight threads on
CPUs `24-31`.  Matched baseline and candidate timings must use the same
reservation and guarded affinity.  A job rejected by this guard is recorded as
a pre-compute topology rejection; it contains no HF or CCSD timing.
