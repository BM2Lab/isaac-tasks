"""Offline parity against the exported, pinned native controller oracle."""
import json
from hashlib import sha256
from pathlib import Path

import torch

from .libero_osc import LiberoOscController, OscState


def check_reference(path, *, device="cpu", atol=2e-5, rtol=2e-6):
    path = Path(path)
    reference = json.loads(path.read_text())
    if reference["schema_version"] != 1:
        raise ValueError("Unsupported reference schema")
    # Full native exports must be complete and match their receipts. The small
    # Git regression subset intentionally excludes the separately stored MJCF.
    if "model" in reference:
        receipt = json.loads((path.parent / "receipt.json").read_text())
        if sha256(path.read_bytes()).hexdigest() != receipt["reference_sha256"]:
            raise ValueError("Reference receipt checksum mismatch")
        xml = path.parent / "model.xml"
        if sha256(xml.read_bytes()).hexdigest() != reference["model"]["xml_sha256"]:
            raise ValueError("Model XML checksum mismatch")
    results = []
    for case in reference["cases"]:  # offline cases, not environment hot path
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float64).unsqueeze(0)
        state = OscState(**{key: tensor(value) for key, value in case["state"].items()})
        controller = LiberoOscController(state, tensor(case["initial_joint"]),
            reference["torque_lower"], reference["torque_upper"], uncoupling=case["uncoupling"])
        controller.set_goal(tensor(case["warm_action"]), state)
        controller.set_goal(tensor(case["action"]), state)
        if case["goal_override"]:
            controller.goal_rotation.copy_(tensor(case["goal_rotation"]))
        actual = controller.compute(state)
        actual.update(goal_position=controller.goal_position, goal_rotation=controller.goal_rotation)
        errors = {}
        normalized_errors = {}
        passed = True
        for key in ("goal_position", "goal_rotation", "raw_torque", "torque", "nullspace"):
            expected = tensor(case[key])
            errors[key] = float((actual[key]-expected).abs().max().cpu())
            normalized_errors[key] = float(((actual[key]-expected).abs() / (atol+rtol*expected.abs())).max().cpu())
            passed = passed and bool(torch.allclose(actual[key], expected, atol=atol, rtol=rtol))
        results.append({"case": case["name"], "passed": passed, "max_absolute_error": errors,
                        "max_error_over_tolerance": normalized_errors})
    return {"passed": bool(results) and all(row["passed"] for row in results),
            "device": device, "dtype": "float64", "atol": atol, "rtol": rtol, "cases": results}
