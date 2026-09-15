# 单卡 A100 外部 CCSD baseline 接入门禁

`external_result_gate.py` 只读取已有结果和证据。它不会连接 MTU，也没有
`sbatch`、shell 或子进程执行路径。两个命令默认把 JSON 打到标准输出；只有显式
给出 `--output` 或 `--manifest` 时才创建一个新文件，而且拒绝覆盖已有文件。

## 分类边界

- ByteQC canonical-four-index 是 **S/L1 protocol-adjacent candidate**。WATER2
  必须与一份可验证的 GPU4PySCF canonical oracle 在显式误差阈值内一致，才允许
  生成 WATER4 的 Python runner argv。现有结果路径没有计时内的 normal
  checkpoint、final full-space residual，也没有证明与 V1 完全相同的轨道文件，
  且只有 commit 与选定文件 SHA、没有 immutable source-tree content SHA。因此
  它保持 L1，不计入 V1 完整 post-HF 的正式加速倍数。
- GANSU RI B-native 是 **I/L2 diagnostic**。它的公开计时边界是 RHF 到
  RI-RCCSD，且 release binary 没有可复现的 source-build attestation；它不能成为
  V1 post-HF 正式加速基线。当前 pinned v2 result 只有 callback 推断收敛，未
  提供 final observed residual/update norm；convergence gate 因此永久拒绝它从
  WATER2 晋级。runner 也只支持 WATER2。若将来同时补齐 final convergence
  measurement 与 WATER4 接口，必须用新 schema 和 runner SHA 重新跑 WATER2，
  不能沿用旧结果授权新代码。

无论软件类别，只要缺失正常 checkpoint 或最终 full-space residual，
`formal_v1_post_hf_comparability.eligible` 就保持 `false`。
此外，当前 ByteQC v2 runner 与 scheduler receipt 没有独立绑定 sealed source tree；
result 自报的任意 `source_tree_sha256` 不能自证来源。即使手工补齐该字段、
checkpoint、residual 和 orbital SHA，本 v1 capability 仍无条件给出
`immutable source tree not independently receipt-bound`，保持 formal `false`。
未来只能由新 schema 接入独立签名或 sealed source receipt，不能复用当前字段解锁。

## 终态 scheduler 证据

审计器兼容已有 `byteqc_job_*_audit.json` 和 `gansu_job_*_audit.json` 的层级。
作业结束后，证据必须提供可核对的 Job ID、`COMPLETED`、`0:0` 和实际节点。
最清楚的补充形式是：

```json
{
  "terminal_evidence": {
    "job_id": "63069",
    "state": "COMPLETED",
    "exit_code": "0:0",
    "node": "compute-1-2"
  }
}
```

原有 source SHA ledger 必须保留。ByteQC 使用顶层 `source_sha256`；GANSU 使用
`files.runtime_assets`。结果中的 Slurm Job ID、host、runner/input/native SHA、
A100 UUID、PCIe Gen4 x16、NUMA 3、CPU 24–31、HBM 与 RSS 都会再次核对。
PENDING、CANCELLED、缺 ExitCode、缺节点、NaN、Infinity 或任何不一致都会
fail closed。所有输入都按严格 JSON 解析，拒绝 `NaN`、`Infinity` 和
`-Infinity` 这类非标准常量；每个几何行必须恰好是一个合法原子符号和三个有限
JSON 数字。

## 审计 WATER2

以下 oracle 可以直接使用项目的 completed GPU4PySCF canonical WATER2 JSON；
它自身还必须有 immutable source snapshot、计时内 checkpoint、计时内
full-space residual，且 residual `<=1e-6`。能量阈值必须显式给出，并且不能松于
项目上限 `1e-4 Eh`。

```bash
python baseline-comparison/external_result_gate.py audit \
  --software byteqc \
  --result /path/to/water2-byteqc.json \
  --scheduler-evidence /path/to/byteqc-job-terminal.json \
  --cases /path/to/cases.json \
  --runner-capabilities baseline-comparison/byteqc-runner-capabilities.v1.json \
  --oracle /path/to/water2-gpu4pyscf-canonical.json \
  --energy-tolerance 1e-8 \
  --output /new/path/byteqc-water2-decision.json
```

GANSU 使用 `--software gansu` 和
`gansu-runner-capabilities.v1.json`。当前 v2 结果会保留为 I/L2 诊断记录，但
由于没有 final observed convergence value，`water2_gate_passed=false` 且
`w4_eligible=false`；runner capability 的 WATER4 拒绝是第二个独立阻断项。

## 生成 WATER4 dry-run manifest

planner 不信任 decision 里保存的 pass/fail 布尔值。它读取 decision 保存的精确
software/tolerance 参数和 evidence paths，重新执行完整 audit，再比较新旧 decision
的完整 payload digest 与所有语义字段；只有重算结果通过且完全一致时才继续。
它还会重新计算每个输入文件以及 runner 的 SHA-256。任何文件在 audit 后变化，
或手工篡改 decision 的 flags/reasons，都会拒绝生成 argv。

```bash
python baseline-comparison/external_result_gate.py plan-water4 \
  --decision /path/to/byteqc-water2-decision.json \
  --runner /path/to/frozen/byteqc_benchmark.py \
  --python /path/to/frozen/venv/bin/python \
  --run-id byteqc-water4-a \
  --output-dir /future/result/directory \
  --manifest /new/path/byteqc-water4-plan.json
```

合格 manifest 中仅有 `job.benchmark_argv`、固定 SHA 环境变量和资源合同；
`dry_run=true`、`would_submit=false`、`submission_argv=null`。不合格 manifest
没有 `job` 字段，并在 `refusal_reasons` 中列出全部阻断项。

## 本地验证

```bash
python -m pytest -q baseline-comparison/tests/test_external_result_gate.py
```
