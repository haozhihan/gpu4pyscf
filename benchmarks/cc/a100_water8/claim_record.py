"""Claim-record helpers for the fixed A100 CCSD benchmark.

The benchmark driver writes experiment records, while this module produces the
smaller claim record used for publication or a performance report.  Validation
is deliberately conservative: a missing value is reported as missing and is
never turned into a passing claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).with_name("claim-record.schema.json")


def _schema_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return SCHEMA_PATH


def _type_ok(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)


def _deterministic_validate(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Small deterministic validator for environments without ``jsonschema``.

    It covers every constraint used by the local claim schema, including the
    conditional rules.  It intentionally does not pretend to be a general JSON
    Schema implementation.
    """

    errors: list[str] = []
    required = schema.get("required", [])
    for key in required:
        if key not in record:
            errors.append(f"missing required property: {key}")
    allowed = set(schema.get("properties", {}))
    for key in record:
        if key not in allowed and schema.get("additionalProperties") is False:
            errors.append(f"additional property: {key}")
    for key, spec in schema.get("properties", {}).items():
        if key not in record:
            continue
        value = record[key]
        if "type" in spec and not _type_ok(value, spec["type"]):
            errors.append(f"{key}: expected {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key}: value is outside enum")
        if isinstance(value, str) and "minLength" in spec and len(value) < spec["minLength"]:
            errors.append(f"{key}: string is empty")
        if isinstance(value, list) and "minItems" in spec and len(value) < spec["minItems"]:
            errors.append(f"{key}: too few items")

    run_spec = schema.get("$defs", {}).get("run", {})
    for name in ("baseline", "candidate"):
        run = record.get(name)
        if not isinstance(run, dict):
            continue
        for key in run_spec.get("required", []):
            if key not in run:
                errors.append(f"{name}: missing required property: {key}")
        if run_spec.get("additionalProperties") is False:
            for key in run:
                if key not in run_spec.get("properties", {}):
                    errors.append(f"{name}: additional property: {key}")
        for key, spec in run_spec.get("properties", {}).items():
            if key not in run:
                continue
            value = run[key]
            if "type" in spec and not _type_ok(value, spec["type"]):
                errors.append(f"{name}.{key}: expected {spec['type']}")
            if spec.get("minimum") is not None and isinstance(value, (int, float)) and value < spec["minimum"]:
                errors.append(f"{name}.{key}: below minimum")
    speed = record.get("speedup")
    if isinstance(speed, dict):
        for key in ("value", "formula"):
            if key not in speed:
                errors.append(f"speedup: missing required property: {key}")
        if "value" in speed and (not _type_ok(speed["value"], "number") or speed["value"] <= 0):
            errors.append("speedup.value: must be positive")
    acc = record.get("accuracy")
    if isinstance(acc, dict):
        for key in ("contract_id", "passed", "observables", "same_eri_representation"):
            if key not in acc:
                errors.append(f"accuracy: missing required property: {key}")
        observables = acc.get("observables")
        if isinstance(observables, list):
            if not observables:
                errors.append("accuracy.observables: too few items")
            for i, obs in enumerate(observables):
                if not isinstance(obs, dict):
                    errors.append(f"accuracy.observables[{i}]: expected object")
                    continue
                for key in ("name", "error", "threshold", "unit", "passed"):
                    if key not in obs:
                        errors.append(f"accuracy.observables[{i}]: missing {key}")
                for key in ("error", "threshold"):
                    if key in obs and (not _type_ok(obs[key], "number") or obs[key] < 0):
                        errors.append(f"accuracy.observables[{i}].{key}: invalid number")

    acceptance = record.get("acceptance_table")
    if acceptance == "S":
        if record.get("same_scientific_input") is not True:
            errors.append("S claims require same_scientific_input=true")
        if record.get("same_timing_boundary") is not True:
            errors.append("S claims require same_timing_boundary=true")
        if isinstance(acc, dict):
            if acc.get("passed") is not True:
                errors.append("S claims require accuracy.passed=true")
            if acc.get("same_eri_representation") is not True:
                errors.append("S claims require same_eri_representation=true")
    if record.get("method_class") == "precision-recovery" and acceptance == "S":
        if isinstance(acc, dict) and acc.get("cleanup_included_in_candidate_time") is not True:
            errors.append("precision-recovery S claims require cleanup in candidate time")
    if acceptance in {"I", "A"} and "approximation_parameters" not in record:
        errors.append(f"{acceptance} claims require approximation_parameters")
    return errors


def validate_claim_record(record: dict[str, Any], schema_path: str | Path | None = None) -> dict[str, Any]:
    """Validate one claim record and return ``valid``, ``errors`` and ``mode``."""

    schema_file = _schema_path(schema_path)
    schema = json.loads(schema_file.read_text())
    try:
        import jsonschema  # type: ignore
    except ImportError:
        errors = _deterministic_validate(record, schema)
        return {"valid": not errors, "errors": errors, "mode": "deterministic"}
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(error.message for error in validator.iter_errors(record))
    return {"valid": not errors, "errors": errors, "mode": "jsonschema"}


def make_claim_record(
    *,
    claim_id: str,
    method_class: str,
    acceptance_table: str,
    timing_scope: str,
    same_scientific_input: bool,
    same_timing_boundary: bool,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    speedup: float,
    formula: str,
    contract_id: str,
    accuracy_observables: list[dict[str, Any]],
    accuracy_passed: bool,
    same_eri_representation: bool,
    approximation_parameters: dict[str, Any] | None = None,
    cleanup_included: bool | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a schema-shaped claim record from already checked evidence."""

    record: dict[str, Any] = {
        "claim_id": claim_id,
        "evidence_state": "local_measured",
        "method_class": method_class,
        "acceptance_table": acceptance_table,
        "timing_scope": timing_scope,
        "same_scientific_input": bool(same_scientific_input),
        "same_timing_boundary": bool(same_timing_boundary),
        "baseline": baseline,
        "candidate": candidate,
        "speedup": {"value": float(speedup), "formula": formula},
        "accuracy": {
            "contract_id": contract_id,
            "passed": bool(accuracy_passed),
            "same_eri_representation": bool(same_eri_representation),
            "observables": accuracy_observables,
        },
    }
    if cleanup_included is not None:
        record["accuracy"]["cleanup_included_in_candidate_time"] = bool(cleanup_included)
    if approximation_parameters is not None:
        record["approximation_parameters"] = approximation_parameters
    if limitations:
        record["limitations"] = list(limitations)
    return record
