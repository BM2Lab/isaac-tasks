"""Run numerical P0 checks without launching Kit or importing MuJoCo."""
import argparse
import json
from isaac_tasks.controllers.libero_osc_checks import check_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = check_reference(args.reference, device=args.device)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
