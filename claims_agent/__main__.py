"""Command line: ``python -m claims_agent "claim text" [--reported-on YYYY-MM-DD] [--trace]``"""
from __future__ import annotations

import argparse
import json
from datetime import date

from .graph import ClaimsAgent


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess one insurance claim.")
    ap.add_argument("text", help="Free-text claim declaration (FR or EN)")
    ap.add_argument("--reported-on", type=date.fromisoformat, default=None)
    ap.add_argument("--trace", action="store_true", help="Include the step-by-step trace")
    args = ap.parse_args()
    result = ClaimsAgent().assess(args.text, args.reported_on).model_dump(mode="json")
    if not args.trace:
        result.pop("trace")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
