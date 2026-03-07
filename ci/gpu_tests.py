import glob
import json
import os
import sys
import threading
import time

import requests

API_BASE = "https://console.vast.ai/api/v0"
API_KEY = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_AI_API_KEY", "")
IMAGE = "ghcr.io/homebrewml/heavyball-ci:latest"


def _detect_repo_and_branch():
    repo = os.environ.get("REPO_URL")
    branch = os.environ.get("BRANCH")
    if repo and branch:
        return repo, branch
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.isfile(event_path):
        try:
            with open(event_path) as f:
                event = json.load(f)
            head = event.get("pull_request", {}).get("head", {})
            if not repo:
                repo = head.get("repo", {}).get("clone_url")
            if not branch:
                branch = head.get("ref")
        except (json.JSONDecodeError, KeyError):
            pass
    return (
        repo or "https://github.com/HomebrewML/HeavyBall",
        branch or os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", "main"),
    )


REPO_URL, BRANCH = _detect_repo_and_branch()
TIMEOUT = 1800
POLL_INTERVAL = 5
MAX_DPH = 0.20
STUCK_TIMEOUT = 30
MAX_RETRIES = 3


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
        "limit": n * 4,
    }
    r = api("POST", "/bundles/", json=query)
    offers = r.json().get("offers", [])
    if not offers:
        print("No GPU offers found matching criteria", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(offers)} offers, need {n}")
    return offers


SELF_DESTRUCT_TIMEOUT = 1800

ONSTART_TEMPLATE = """#!/bin/bash
timeout {timeout} bash -c '
export PIP_BREAK_SYSTEM_PACKAGES=1 &&
cd / && git clone --depth 1 -b {branch} {repo} /w &&
cd /w && pip install -e ".[dev]" -q --break-system-packages 2>&1 &&
python -m pytest {test} --tb=short -q 2>&1; echo HEAVYBALL_EXIT=$?
'
"""


def create_instance(offer_id, test_file):
    payload = {
        "client_id": "me",
        "image": IMAGE,
        "disk": 16,
        "onstart": ONSTART_TEMPLATE.format(
            timeout=SELF_DESTRUCT_TIMEOUT,
            branch=BRANCH,
            repo=REPO_URL,
            test=test_file,
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
    try:
        r = api("GET", "/instances", params={"owner": "me"})
        return {inst["id"]: inst for inst in r.json().get("instances", [])}
    except Exception:
        return {}


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
        return True
    except Exception:
        return False


def destroy_all(extra_ids=()):
    to_destroy = set(extra_ids)
    to_destroy.update(get_instances())

    while to_destroy:
        for iid in to_destroy:
            destroy(iid)
        time.sleep(3)
        to_destroy = set(get_instances())


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
        "file": test_file,
        "status": status,
        "exit_code": exit_code,
        "duration": _instance_elapsed(inst),
        "log": log or "",
    }


def parse_exit_code(log):
    for line in reversed(log.splitlines()):
        if line.startswith("HEAVYBALL_EXIT="):
            return int(line.split("=")[1])
    return None


_ICONS = {"pass": "+", "fail": "-", "error": "!", "timeout": "?"}


def _log(icon, test_file, msg):
    print(f"  [{icon}] {test_file}: {msg}")


def _print_progress(results, total):
    p = sum(1 for r in results.values() if r["status"] == "pass")
    f = sum(1 for r in results.values() if r["status"] == "fail")
    e = sum(1 for r in results.values() if r["status"] in ("error", "timeout"))
    print(f"  {len(results)}/{total} done ({p} pass, {f} fail, {e} err), {total - len(results)} pending")


def _is_infra_error(inst):
    msg = (inst.get("status_msg") or "").lower()
    return "error response from daemon" in msg or "oci runtime" in msg or "not running" in msg or "error writing" in msg


def _error_result(test_file, log=""):
    return {"file": test_file, "status": "error", "exit_code": -1, "duration": 0, "log": log}


def _try_recycle(test_file, spare_offers, instance_map, created_at, pending):
    while spare_offers:
        offer = spare_offers.pop(0)
        try:
            new_iid = create_instance(offer["id"], test_file)
        except Exception:
            continue
        if new_iid:
            instance_map[new_iid] = test_file
            created_at[new_iid] = time.time()
            pending.add(new_iid)
            return True
    return False


def _recycle_or_fail(iid, test_file, reason, retries, spare_offers, instance_map, created_at, pending, results):
    pending.discard(iid)
    destroy(iid)
    retries[test_file] = retries.get(test_file, 0) + 1
    attempt = retries[test_file]
    if attempt <= MAX_RETRIES and _try_recycle(test_file, spare_offers, instance_map, created_at, pending):
        _log("!", test_file, f"{reason}, retry {attempt}/{MAX_RETRIES}")
    else:
        give_up = "max retries" if attempt > MAX_RETRIES else "no spare offers"
        results[iid] = _error_result(test_file, reason)
        _log("!", test_file, f"{reason}, {give_up}")


def wait_and_collect(instance_map, spare_offers, timeout=TIMEOUT):
    results = {}
    pending = set(instance_map.keys())
    total = len(instance_map)
    deadline = time.time() + timeout
    created_at = {iid: time.time() for iid in instance_map}
    retries = {}

    while pending and time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        all_instances = get_instances()

        to_recycle = []
        to_fetch = {}
        stuck_candidates = set()

        for iid in list(pending):
            inst = all_instances.get(iid)
            if inst is None:
                to_recycle.append((iid, "disappeared"))
                continue
            if _is_infra_error(inst):
                to_recycle.append((iid, f"docker error: {inst.get('status_msg', '')}"))
                continue

            status = inst.get("actual_status", "")
            done = status in ("exited", "error", "offline")
            age = time.time() - created_at[iid]

            if done or (status == "running" and _instance_elapsed(inst) >= 60):
                to_fetch[iid] = (inst, done)
            elif age >= STUCK_TIMEOUT:
                to_fetch[iid] = (inst, False)
                stuck_candidates.add(iid)

        ctx = (retries, spare_offers, instance_map, created_at, pending, results)
        for iid, reason in to_recycle:
            _recycle_or_fail(iid, instance_map[iid], reason, *ctx)

        logs = {}
        threads = [threading.Thread(target=lambda i=iid: logs.__setitem__(i, get_logs(i))) for iid in to_fetch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for iid, (inst, done) in to_fetch.items():
            log = logs.get(iid, "")
            ec = parse_exit_code(log)
            if done or ec is not None:
                result = _make_result(instance_map[iid], ec, inst, log)
                results[iid] = result
                pending.discard(iid)
                destroy(iid)
                _log(_ICONS[result["status"]], instance_map[iid], f"{result['status']} ({result['duration']}s)")
            elif iid in stuck_candidates and not log.strip():
                age = int(time.time() - created_at[iid])
                _recycle_or_fail(iid, instance_map[iid], f"no logs after {age}s", *ctx)

        _print_progress(results, total)

    for iid in pending:
        log = get_logs(iid)
        results[iid] = {
            "file": instance_map[iid],
            "status": "timeout",
            "exit_code": -1,
            "duration": timeout,
            "log": (log[-4000:] if log else "Timed out"),
        }
        destroy(iid)
        _log("?", instance_map[iid], f"timeout ({timeout}s)")

    return list(results.values())


def main():
    if not API_KEY:
        print("Set VASTAI_API_KEY or VAST_AI_API_KEY", file=sys.stderr)
        sys.exit(1)
    test_files = sorted(glob.glob("test/test_*.py"))
    if not test_files:
        print("No test files found", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(test_files)} test files")

    all_offers = find_offers(len(test_files))
    offers = all_offers[: len(test_files)]
    spare_offers = list(all_offers[len(test_files) :])
    if len(offers) < len(test_files):
        print(f"Warning: only {len(offers)} offers for {len(test_files)} tests, some skipped")
        test_files = test_files[: len(offers)]
    print(f"{len(offers)} primary, {len(spare_offers)} spare offers")

    instance_map = {}
    try:
        for test_file, offer in zip(test_files, offers):
            try:
                iid = create_instance(offer["id"], test_file)
            except Exception as e:
                print(f"  Failed to create for {test_file}: {e}")
                continue
            if iid:
                instance_map[iid] = test_file
            else:
                print(f"  Failed to create for {test_file}")

        if not instance_map:
            print("No instances created", file=sys.stderr)
            sys.exit(1)

        print(f"\nWaiting for {len(instance_map)} instances (timeout={TIMEOUT}s)...")
        results = wait_and_collect(instance_map, spare_offers)
    finally:
        print("\nCleaning up...")
        destroy_all(instance_map)

    with open("gpu-test-results.json", "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
    print(f"\nResults: {passed} passed, {failed} failed, {errors} errors/timeouts")
    for r in sorted(results, key=lambda x: x["file"]):
        _log(_ICONS[r["status"]], r["file"], f"{r['status']} ({r['duration']}s)")

    sys.exit(0 if failed + errors == 0 else 1)


if __name__ == "__main__":
    main()
