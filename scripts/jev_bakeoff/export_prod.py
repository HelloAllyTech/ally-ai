"""Phase 0 export: pull already-judged sessions from prod, read-only, via SSM.

Test organizations ONLY (tenants.isTestOrganization): internal, demo and QA
tenants. No real learner's session is exported, because the next step sends it
to TypeSafe and session content is PHI-adjacent.

Runs `export_query.js` inside the prod ally-be core container in pages (SSM
returns at most 24KB of stdout), and writes one JSON file per judge to the data
directory. The data stays on this machine: it is NOT written into the repo.

    python scripts/jev_bakeoff/export_prod.py --per-stratum 25 --since-days 60

Requires AWS creds from ~/projects/Ally/aws_session.env.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("JEV_BAKEOFF_DATA", Path.home() / ".cache" / "ally-jev-bakeoff"))
CLUSTER = "ally-prd-mb-ecs-cluster"
SERVICE = "ally-prd-svc-core"
REGION = "ap-south-1"
MAX_STDOUT = 24000


def aws_env() -> dict:
    env = dict(os.environ, AWS_REGION=REGION)
    session = Path.home() / "projects" / "Ally" / "aws_session.env"
    for line in session.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def aws(env: dict, *args: str) -> str:
    return subprocess.run(
        ["aws", *args], env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def instance_id(env: dict) -> str:
    task = aws(env, "ecs", "list-tasks", "--cluster", CLUSTER, "--service-name", SERVICE,
               "--query", "taskArns[0]", "--output", "text")
    ci = aws(env, "ecs", "describe-tasks", "--cluster", CLUSTER, "--tasks", task,
             "--query", "tasks[0].containerInstanceArn", "--output", "text")
    return aws(env, "ecs", "describe-container-instances", "--cluster", CLUSTER,
               "--container-instances", ci, "--query",
               "containerInstances[0].ec2InstanceId", "--output", "text")


def run_page(env: dict, iid: str, params: dict):
    """One SSM round trip. Returns the decoded JSON, or None if truncated."""
    js = base64.b64encode((HERE / "export_query.js").read_bytes()).decode()
    arg = base64.b64encode(json.dumps(params).encode()).decode()
    commands = [
        "set -e",
        f"echo {js} | base64 -d > /tmp/bakeoff_query.js",
        "CID=$(docker ps -q -f name=cntr-core | head -1)",
        "docker cp /tmp/bakeoff_query.js $CID:/tmp/bakeoff_query.js",
        f"docker exec -w /app $CID node /tmp/bakeoff_query.js {arg}",
        "docker exec $CID rm -f /tmp/bakeoff_query.js; rm -f /tmp/bakeoff_query.js",
    ]
    pfile = DATA_DIR / ".ssm_params.json"
    pfile.write_text(json.dumps({"commands": commands}))
    cmd = aws(env, "ssm", "send-command", "--instance-ids", iid, "--document-name",
              "AWS-RunShellScript", "--parameters", f"file://{pfile}",
              "--query", "Command.CommandId", "--output", "text")
    for _ in range(60):
        time.sleep(3)
        try:
            status = aws(env, "ssm", "get-command-invocation", "--command-id", cmd,
                         "--instance-id", iid, "--query", "Status", "--output", "text")
        except subprocess.CalledProcessError:
            continue
        if status in ("Pending", "InProgress", "Delayed"):
            continue
        out = aws(env, "ssm", "get-command-invocation", "--command-id", cmd,
                  "--instance-id", iid, "--query", "StandardOutputContent", "--output", "text")
        if status != "Success":
            err = aws(env, "ssm", "get-command-invocation", "--command-id", cmd,
                      "--instance-id", iid, "--query", "StandardErrorContent", "--output", "text")
            raise RuntimeError(f"SSM {status}: {err[-500:]}")
        if "BAKEOFF_B64:" not in out or ":END" not in out:
            return None  # truncated at 24KB: caller halves the page
        b64 = out.split("BAKEOFF_B64:", 1)[1].split(":END", 1)[0]
        return json.loads(gzip.decompress(base64.b64decode(b64)))
    raise TimeoutError(f"SSM command {cmd} did not finish")


def fetch_sessions(env, iid, kind, pin, ids, page=8):
    out, i = [], 0
    while i < len(ids):
        chunk = ids[i : i + page]
        got = run_page(env, iid, {"mode": "sessions", "kind": kind, "ids": chunk, **pin})
        if got is None:
            if page == 1:
                print(f"  skip {chunk[0]}: one session exceeds the SSM output cap", file=sys.stderr)
                i += 1
                continue
            page = max(1, page // 2)
            continue
        out.extend(got)
        i += len(chunk)
        print(f"  {kind}: {len(out)}/{len(ids)} sessions", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=25)
    ap.add_argument("--since-days", type=int, default=60)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = aws_env()
    iid = instance_id(env)
    print(f"instance {iid}; writing to {DATA_DIR}", file=sys.stderr)

    plan = run_page(env, iid, {"mode": "plan", "sinceDays": args.since_days,
                               "perStratum": args.per_stratum})
    if plan is None:
        raise RuntimeError("plan output exceeded the SSM cap; lower --per-stratum")

    scenario_ids = sorted({s["scenario_id"] for k in plan.values() for s in k["sessions"]
                           if s["scenario_id"] is not None})
    personas = {}
    for i in range(0, len(scenario_ids), 5):
        got = run_page(env, iid, {"mode": "personas", "scenarioIds": scenario_ids[i : i + 5]})
        if got is None:  # one very long persona: fetch singly, skip if still too big
            for sid in scenario_ids[i : i + 5]:
                got1 = run_page(env, iid, {"mode": "personas", "scenarioIds": [sid]})
                personas.update(got1 or {})
        else:
            personas.update(got)

    for kind, p in plan.items():
        pin = {"judgeModel": p["judgeModel"], "judgePromptVersion": p["judgePromptVersion"]}
        meta = {s["id"]: s for s in p["sessions"]}
        rows = fetch_sessions(env, iid, kind, pin, list(meta))
        for r in rows:
            m = meta[r["id"]]
            r.update(language=m["language"], llm_model=m["llm_model"],
                     persona=personas.get(str(m["scenario_id"]), ""))
        # export_query.js only ever returns test-organization sessions; the
        # marker lets score.py refuse an export made before that restriction.
        (DATA_DIR / f"{kind}.json").write_text(
            json.dumps({**pin, "scope": "test_organizations_only", "sessions": rows})
        )
        print(f"{kind}: pinned {pin['judgeModel']} / {pin['judgePromptVersion']}, "
              f"{len(rows)} sessions exported", file=sys.stderr)


if __name__ == "__main__":
    main()
