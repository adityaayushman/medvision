"""Publish a local evaluation result to the live Research Workspace.

ml/artifacts/ (where evaluate_checkpoint/evaluate_ensemble/
evaluate_bbox_checkpoint/evaluate_segmenter/evaluate_localized_pipeline all
write their metrics.json) is gitignored and never reaches the deployed
backend -- Render's Dockerfile only copies backend/model, the curated
inference checkpoints. This script is the "publish" step: it takes an
existing metrics.json (no new metrics computation) plus run metadata and
POSTs it to POST /api/research/runs, so the result is visible in the
dashboard's Research page instead of living only on whatever machine
trained it.

Usage:
    # 1) log in once to get a token (admin or researcher role)
    python ml/scripts/log_experiment.py login \
        --backend-url https://medchron-backend.onrender.com \
        --email you@example.com --password ...

    # 2) publish a result
    python ml/scripts/log_experiment.py publish \
        --backend-url https://medchron-backend.onrender.com \
        --token <jwt from step 1> \
        --kind classification --modality mammography --backbone efficientnet_b0 \
        --label "CBIS-DDSM cropped-patch" \
        --metrics-file ml/artifacts/mammography_cbis_cropped/metrics.json \
        --notes "71.1% acc / 0.785 AUC -- real win, but needs an already-cropped input"
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{exc.code} {exc.reason}: {detail}") from exc


def cmd_login(args: argparse.Namespace) -> None:
    result = _post(
        f"{args.backend_url.rstrip('/')}/api/auth/login",
        {"email": args.email, "password": args.password},
    )
    print(result["access_token"])


def cmd_publish(args: argparse.Namespace) -> None:
    with open(args.metrics_file, encoding="utf-8") as fh:
        metrics = json.load(fh)

    result = _post(
        f"{args.backend_url.rstrip('/')}/api/research/runs",
        {
            "kind": args.kind,
            "modality": args.modality,
            "backbone": args.backbone,
            "label": args.label,
            "metrics": metrics,
            "notes": args.notes,
        },
        token=args.token,
    )
    print(f"Published run #{result['id']}: {result['label']} ({result['kind']}/{result['modality']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", help="Get a JWT to use with 'publish'")
    login.add_argument("--backend-url", required=True)
    login.add_argument("--email", required=True)
    login.add_argument("--password", required=True)
    login.set_defaults(func=cmd_login)

    publish = sub.add_parser("publish", help="Publish a metrics.json as an experiment run")
    publish.add_argument("--backend-url", required=True)
    publish.add_argument("--token", required=True)
    publish.add_argument("--kind", required=True,
                          choices=["classification", "bbox_regression", "segmentation", "ensemble"])
    publish.add_argument("--modality", required=True)
    publish.add_argument("--backbone", required=True)
    publish.add_argument("--label", required=True)
    publish.add_argument("--metrics-file", required=True)
    publish.add_argument("--notes", default=None)
    publish.set_defaults(func=cmd_publish)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
