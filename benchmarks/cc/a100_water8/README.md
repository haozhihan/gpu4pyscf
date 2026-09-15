# A100 WATER27 CCSD harness

This directory contains a reproducible harness for water2 and water4 validation,
the `H2O8d2d` performance target, and the independent `H2O8s4` accuracy
holdout. The first three geometries are carried over from the prior MTU case
file. The holdout is copied from the official
[GMTKN55 repository](https://github.com/grimme-lab/GMTKN55) at the revision and
file SHA-256 recorded in `cases.json`. Coordinates are in angstroms and all
cases use spherical `cc-pVTZ`.

The driver supports four explicit method labels:

* `canonical`: GPU4PySCF canonical CCSD.
* `fno`: MP2 frozen-natural-orbital CCSD with the recorded Delta-MP2 value.
* `rr_canonical`: RRCCSD using the dense canonical residual for validation.
* `thc_canonical`: the dense two-level THC validation surrogate.

The latter two records state their validation status in every result and must
not be used to claim reduced-scaling production performance.  No speedup claim
is made by this harness.

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
an rsync deployment without `.git` records the same deterministic content
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
the Delta-MP2-corrected CCSD correlation error, active-space residual, HBM and
host RSS, and the shared-bundle counterpoise error. It checks the algebraic
identity `Delta-MP2 = E_MP2(full) - E_MP2(FNO)` and the corresponding corrected
CCSD energies. The water8 candidate list is emitted only when all seven
threshold records are present and both the first loose-to-tight passing point
and its immediately tighter neighbour independently satisfy `1e-4 Eh`,
`0.02 kcal/mol`, residual, resource, exact-orbital, immutable-source, hardware,
and positive water4 speedup gates. This is a development probe; final timing
still uses three interleaved fresh-process baseline/candidate pairs.

Formal timings should run from an immutable deployment made by
`deploy_mtu_snapshot.sh`. It hashes the source files using the same function as
the benchmark, rsyncs to a staging directory, verifies the hash on MTU, then
atomically installs a read-only `snapshots/<sha256>/source` tree. Set
`CCSD_SOURCE_ROOT` to the printed path and `PYTHONDONTWRITEBYTECODE=1` for the
Slurm job. Because shared libraries are recorded separately from the source
digest, deployment also compares the requested runtime bundle against every
sealed `.so` by file set, byte count, and SHA-256 before reusing an existing
snapshot. A same-source snapshot with a different rebuilt runtime fails closed.
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

The decision has separate `correctness_passed` and `performance_eligible`
fields. Numerical parity can pass while performance remains false. Performance
eligibility additionally requires the provider's release flag, a complete
`_VHFOpt` setup-transfer audit, and no unresolved transfer operations. This is
deliberately fail-closed; a successful correctness run does not by itself
authorize water2/water4/water8 timing claims. Run the immutable-snapshot gate
through `run_mtu_gint_gate.sbatch`:

```bash
sbatch --parsable \
  --export="ALL,CCSD_TASK_ROOT=${CCSD_TASK_PATH},CCSD_SOURCE_ROOT=${CCSD_SNAPSHOT_PATH},CCSD_EXPECTED_DEPLOYMENT_PROFILE=candidate" \
  run_mtu_gint_gate.sbatch
```

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
