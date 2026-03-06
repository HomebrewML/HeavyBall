#!/usr/bin/env python3
"""Run GPU tests on vast.ai instances in parallel. Each test file gets its own instance."""

import glob
import json
import os
import sys
import time

import requests

API_BASE = "https://console.vast.ai/api/v0"
API_KEY = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_AI_API_KEY", "")
IMAGE = "ghcr.io/homebrewml/heavyball-ci:latest"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/HomebrewML/HeavyBall")
BRANCH = os.environ.get("BRANCH", "main")
TIMEOUT = 1800
POLL_INTERVAL = 60
MAX_DPH = 0.20


def api(method, path, **kwargs):
    kwargs.setdefault("params", {})
    kwargs["params"]["api_key"] = API_KEY
    for attempt in range(3):
        r = requests.request(method, f"{API_BASE}{path}", **kwargs)
        if r.status_code != 429:
            break
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return r


def find_offers(n):
    query = {
        "gpu_ram": {"gte": 8},
        "num_gpus": {"eq": 1},
        "inet_down": {"gte": 200},
        "disk_bw": {"gte": 500},
        "dph_total": {"lte": MAX_DPH},
        "cuda_vers": {"gte": 12.0},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "limit": n * 2,
    }
    r = api("POST", "/bundles/", json=query)
    offers = r.json().get("offers", [])
    if not offers:
        print("No GPU offers found matching criteria", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(offers)} offers, need {n}")
    return offers[:n]


SELF_DESTRUCT_TIMEOUT = 1800

ONSTART_TEMPLATE = """#!/bin/bash
_self_destruct() {{ sleep 90; curl -s -X DELETE \
  "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/?api_key=$CONTAINER_API_KEY"; }}
trap _self_destruct EXIT
timeout {timeout} bash -c '
pip install -q opt-einsum numpy pytest hypothesis lightbench 2>&1 &&
cd / && git clone --depth 1 -b {branch} {repo} /w &&
cd /w && pip install -e . --no-deps -q &&
python -m pytest {test} --tb=short -q 2>&1; echo HEAVYBALL_EXIT=$?
'
"""


def create_instance(offer_id, test_file):
    payload = {
        "client_id": "me",
        "image": IMAGE,
        "disk": 16,
        "onstart": ONSTART_TEMPLATE.format(
            timeout=SELF_DESTRUCT_TIMEOUT, branch=BRANCH,
            repo=REPO_URL, test=test_file,
        ),
        "runtype": "ssh_direc ssh_proxy",
    }
    r = api("PUT", f"/asks/{offer_id}/", json=payload)
    rj = r.json()
    if not rj.get("success"):
        return None
    instance_id = rj["new_contract"]
    print(f"  Created instance {instance_id} for {test_file} on offer {offer_id}")
    return instance_id


def get_instances():
    r = api("GET", "/instances", params={"owner": "me"})
    return {inst["id"]: inst for inst in r.json().get("instances", [])}


def get_logs(instance_id):
    try:
        r = api("PUT", f"/instances/request_logs/{instance_id}/")
        rj = r.json()
        url = rj.get("result_url")
        if not url:
            return ""
        for _ in range(10):
            time.sleep(1)
            resp = requests.get(url)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
        return ""
    except Exception:
        return ""


def destroy(instance_id):
    try:
        api("DELETE", f"/instances/{instance_id}/")
    except Exception:
        pass


def destroy_all():
    try:
        for iid in get_instances():
            destroy(iid)
    except Exception:
        pass


def _instance_elapsed(inst):
    start = inst.get("start_date")
    if start:
        return round(time.time() - start)
    return 0


def _make_result(test_file, exit_code, inst, log):
    if exit_code is None:
        status, exit_code = "error", -1
    elif exit_code == 0:
        status = "pass"
    else:
        status = "fail"
    return {
        "file": test_file, "status": status, "exit_code": exit_code,
        "duration": _instance_elapsed(inst),
        "log": log[-4000:] if log else "",
    }


def parse_exit_code(log):
    for line in reversed(log.splitlines()):
        if line.startswith("HEAVYBALL_EXIT="):
            return int(line.split("=")[1])
    return None


def wait_and_collect(instance_map, timeout=TIMEOUT):
    results = {}
    pending = set(instance_map.keys())
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        all_instances = get_instances()

        for iid in list(pending):
            inst = all_instances.get(iid)
            if inst is None:
                results[iid] = {
                    "file": instance_map[iid], "status": "error",
                    "exit_code": -1, "duration": 0, "log": "Instance disappeared",
                }
                pending.discard(iid)
                continue

            status = inst.get("actual_status", "")
            if status in ("exited", "error", "offline"):
                log = get_logs(iid)
                ec = parse_exit_code(log)
                results[iid] = _make_result(instance_map[iid], ec, inst, log)
                pending.discard(iid)
            elif status == "running" and _instance_elapsed(inst) >= 60:
                log = get_logs(iid)
                ec = parse_exit_code(log)
                if ec is not None:
                    results[iid] = _make_result(instance_map[iid], ec, inst, log)
                    pending.discard(iid)

        done = len(instance_map) - len(pending)
        print(f"  Progress: {done}/{len(instance_map)} complete, {int(deadline - time.time())}s remaining")

    for iid in pending:
        log = get_logs(iid)
        results[iid] = {
            "file": instance_map[iid], "status": "timeout",
            "exit_code": -1, "duration": timeout,
            "log": (log[-4000:] if log else "Timed out"),
        }

    return list(results.values())


def main():
    if not API_KEY:
        print("Set VASTAI_API_KEY or VAST_AI_API_KEY", file=sys.stderr)
        sys.exit(1)
    test_files = sorted(glob.glob("test/test_*.py"))
    if not test_files:
        print("No test files found", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered {len(test_files)} test files")

    offers = find_offers(len(test_files))
    if len(offers) < len(test_files):
        print(f"Warning: only {len(offers)} offers for {len(test_files)} tests, some tests will be skipped")
        test_files = test_files[: len(offers)]

    instance_map = {}
    try:
        for test_file, offer in zip(test_files, offers):
            try:
                iid = create_instance(offer["id"], test_file)
            except Exception as e:
                print(f"  Failed to create instance for {test_file}: {e}")
                continue
            if iid:
                instance_map[iid] = test_file
            else:
                print(f"  Failed to create instance for {test_file}")

        if not instance_map:
            print("No instances created", file=sys.stderr)
            sys.exit(1)

        print(f"\nWaiting for {len(instance_map)} instances (timeout={TIMEOUT}s)...")
        results = wait_and_collect(instance_map)
    finally:
        print("\nCleaning up...")
        destroy_all()

    with open("gpu-test-results.json", "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
    print(f"\nResults: {passed} passed, {failed} failed, {errors} errors/timeouts")

    for r in sorted(results, key=lambda x: x["file"]):
        icon = {"pass": "+", "fail": "-", "error": "!", "timeout": "?"}[r["status"]]
        print(f"  [{icon}] {r['file']} ({r['duration']}s)")

    sys.exit(0 if failed + errors == 0 else 1)


if __name__ == "__main__":
    main()
