"""Distributed training tests: verifies DDP/FSDP produce params within 1 ULP of single-device."""

import os
import tempfile
from pathlib import Path

_test_dir = str(Path(__file__).resolve().parent)
_pp = os.environ.get("PYTHONPATH", "")
if _test_dir not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_test_dir}{os.pathsep}{_pp}" if _pp else _test_dir

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from utils import REPRESENTATIVE_OPTS

import heavyball
from heavyball.utils import clean

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_EXTRA_KWARGS = {"ForeachAdamC": {"max_lr": 0.0025}}
_MODEL_SEED = 42
_DATA_SEED = 0xABCD

# LRA builds one preconditioner over all grads — under FSDP each rank only has a subset
_FSDP_SKIP = {
    "ForeachPSGDLRA": "LRA preconditioner scope differs under FSDP",
    "ForeachDelayedPSGDLRA": "LRA preconditioner scope differs under FSDP",
}

# torch.compile(dynamic=False) specializes on list length → different kernels per rank
_FSDP_NO_COMPILE = {"MSAMLaProp"}

# PSGD uses global RNG for dampening vector V which diverges across FSDP shards;
# allow tolerance-based comparison instead of bitwise identity
_FSDP_PSGD = {n for n in REPRESENTATIVE_OPTS if "PSGD" in n and n not in _FSDP_SKIP}

_SPLIT_OPTS = [n for n in REPRESENTATIVE_OPTS if n not in _FSDP_SKIP]

_INTEGRATION_OPTS = [
    n for n in [
        "ForeachAdamW", "ForeachSOAP", "ForeachMuon", "ForeachPSGDKron",
        "ForeachPurePSGD", "Scion", "ForeachLaProp", "MuonLaProp",
        "ForeachSOAPNAdam", "ForeachCachedPSGDKron",
    ]
    if n in REPRESENTATIVE_OPTS and n not in _FSDP_SKIP
]


def _set_cache(cache_dir, compile_mode="default"):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    # "default" avoids autotuning nondeterminism; None disables compile entirely
    heavyball.utils.compile_mode = compile_mode


def _make_opt(name, params, **kwargs):
    extra = _EXTRA_KWARGS.get(name, {})
    return getattr(heavyball, name)(params, lr=1e-3, **extra, **kwargs)


def _make_model():
    return nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))


def _make_split_model():
    return nn.Sequential(
        nn.Linear(64, 64, bias=False), nn.ReLU(),
        nn.Linear(64, 64, bias=False), nn.ReLU(),
        nn.Linear(64, 64, bias=False),
    )


def _make_integration_model():
    return nn.Sequential(
        nn.LayerNorm(64), nn.Linear(64, 64, bias=False), nn.ReLU(),
        nn.LayerNorm(64), nn.Linear(64, 64, bias=False), nn.ReLU(),
        nn.Linear(64, 64, bias=False), nn.ReLU(),
        nn.Linear(64, 64, bias=False),
    )


def _make_data(dim=32, n=4):
    torch.manual_seed(_DATA_SEED)
    return [torch.randn(4, dim, device="cuda") for _ in range(n)]


def _make_split_data():
    return _make_data(64)


def _make_integration_data():
    return _make_data(64, 8)


def _train(model, opt, data):
    for x in data:
        model(x).mean().backward()
        opt.step()
        opt.zero_grad()


def _save(model, path):
    torch.save([p.detach().cpu() for p in model.parameters()], path)


def _init_dist(rank, world_size, store_path):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world_size), rank=rank, world_size=world_size)
    torch.cuda.set_device(0)


def _ref_worker(rank, opt_name, result_path, cache_dir, compile_mode="default",
                model_fn=_make_model, data_fn=_make_data):
    _set_cache(cache_dir, compile_mode)
    torch.cuda.set_device(0)
    torch.manual_seed(_MODEL_SEED)
    model = model_fn().cuda()
    opt = _make_opt(opt_name, model.parameters())
    _train(model, opt, data_fn())
    _save(model, result_path)
    del opt, model; clean()


def _ddp_worker(rank, world_size, store_path, opt_name, result_path, cache_dir, compile_mode="default"):
    _set_cache(cache_dir, compile_mode)
    _init_dist(rank, world_size, store_path)
    try:
        torch.manual_seed(_MODEL_SEED)
        model = _make_model().cuda()
        ddp = nn.parallel.DistributedDataParallel(model)
        opt = _make_opt(opt_name, ddp.parameters())
        _train(ddp, opt, _make_data())
        if rank == 0:
            _save(model, result_path)
        del opt, ddp, model; clean()
    finally:
        dist.destroy_process_group()


def _fsdp_worker(rank, world_size, store_path, opt_name, result_path, cache_dir,
                 compile_mode="default", model_fn=_make_model, data_fn=_make_data):
    _set_cache(cache_dir, compile_mode)
    _init_dist(rank, world_size, store_path)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        torch.manual_seed(_MODEL_SEED)
        model = model_fn().cuda()
        fsdp = FSDP(model, use_orig_params=True)
        opt = _make_opt(opt_name, fsdp.parameters())
        _train(fsdp, opt, data_fn())
        with FSDP.summon_full_params(fsdp):
            if rank == 0:
                _save(fsdp, result_path)
        del opt, fsdp, model; clean()
    finally:
        dist.destroy_process_group()


def _fsdp2_worker(rank, world_size, store_path, opt_name, result_path, cache_dir):
    _set_cache(cache_dir)
    _init_dist(rank, world_size, store_path)
    try:
        from torch.distributed._composable.fsdp import fully_shard
        torch.manual_seed(_MODEL_SEED)
        model = _make_model().cuda()
        for m in model:
            if isinstance(m, nn.Linear):
                fully_shard(m)
        fully_shard(model)
        opt = _make_opt(opt_name, model.parameters())
        _train(model, opt, _make_data())
        # all ranks must participate in full_tensor (all-gather)
        params = [p.full_tensor().detach().cpu() for p in model.parameters()]
        if rank == 0:
            torch.save(params, result_path)
        del opt, model, params; clean()
    finally:
        dist.destroy_process_group()


class _LazyRefs:
    def __init__(self):
        self._cache = {}

    def get(self, name, compile_mode="default"):
        key = (name, compile_mode)
        if key not in self._cache:
            cache_dir = tempfile.mkdtemp(prefix=f"hb_{name}_")
            f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            f.close()
            mp.spawn(_ref_worker, args=(name, f.name, cache_dir, compile_mode), nprocs=1, join=True)
            self._cache[key] = {"params": torch.load(f.name, weights_only=True), "cache_dir": cache_dir}
            os.unlink(f.name)
        return self._cache[key]

    def __getitem__(self, name):
        return self.get(name)


@pytest.fixture(scope="module")
def reference_params():
    return _LazyRefs()


def _assert_close(ref, result, label, rtol=0, atol=0):
    assert len(ref) == len(result), f"{label}: param count mismatch ({len(ref)} vs {len(result)})"
    for i, (r, d) in enumerate(zip(ref, result)):
        assert r.shape == d.shape, f"{label}: param {i} shape mismatch ({r.shape} vs {d.shape})"
        if torch.equal(r, d):
            continue
        if rtol or atol:
            assert torch.allclose(r, d, rtol=rtol, atol=atol), (
                f"{label}: param {i} diverged (max |diff|={(r - d).abs().max().item():.2e})")
            continue
        lo = torch.nextafter(r, torch.full_like(r, float('-inf')))
        hi = torch.nextafter(r, torch.full_like(r, float('inf')))
        ok = (d >= lo) & (d <= hi)
        if not ok.all():
            n = (~ok).sum().item()
            worst = (r - d).abs().max().item()
            assert False, f"{label}: param {i} diverged beyond 1 ULP ({n} elements, max |diff|={worst:.2e})"


def _run_fsdp_test(opt_name, tmp_path, model_fn, data_fn, label):
    cache_dir = tempfile.mkdtemp(prefix=f"hb_{label}_{opt_name}_")
    ref_path = str(tmp_path / "ref.pt")
    mp.spawn(_ref_worker, args=(opt_name, ref_path, cache_dir, None, model_fn, data_fn), nprocs=1, join=True)
    ref = torch.load(ref_path, weights_only=True)
    result_path = str(tmp_path / "result.pt")
    mp.spawn(
        _fsdp_worker,
        args=(2, str(tmp_path / "store"), opt_name, result_path, cache_dir, None, model_fn, data_fn),
        nprocs=2, join=True,
    )
    tol = dict(rtol=1e-2, atol=1e-4) if opt_name in _FSDP_PSGD else {}
    _assert_close(ref, torch.load(result_path, weights_only=True), f"{label}/{opt_name}", **tol)


@pytest.mark.parametrize("opt_name", REPRESENTATIVE_OPTS)
def test_ddp(opt_name, reference_params, tmp_path):
    info = reference_params[opt_name]
    result_path = str(tmp_path / "result.pt")
    mp.spawn(
        _ddp_worker,
        args=(2, str(tmp_path / "store"), opt_name, result_path, info["cache_dir"]),
        nprocs=2, join=True,
    )
    _assert_close(info["params"], torch.load(result_path, weights_only=True), f"DDP/{opt_name}")


@pytest.mark.parametrize("opt_name", REPRESENTATIVE_OPTS)
def test_fsdp(opt_name, reference_params, tmp_path):
    if opt_name in _FSDP_SKIP:
        pytest.skip(_FSDP_SKIP[opt_name])
    cm = None if opt_name in _FSDP_NO_COMPILE else "default"
    info = reference_params.get(opt_name, cm)
    result_path = str(tmp_path / "result.pt")
    mp.spawn(
        _fsdp_worker,
        args=(2, str(tmp_path / "store"), opt_name, result_path, info["cache_dir"], cm),
        nprocs=2, join=True,
    )
    tol = dict(rtol=1e-2, atol=1e-4) if opt_name in _FSDP_PSGD else {}
    _assert_close(info["params"], torch.load(result_path, weights_only=True), f"FSDP/{opt_name}", **tol)


@pytest.mark.skip(reason="FSDP2 (fully_shard) segfaults with gloo on single GPU")
@pytest.mark.parametrize("opt_name", REPRESENTATIVE_OPTS)
def test_fsdp2(opt_name, reference_params, tmp_path):
    info = reference_params[opt_name]
    result_path = str(tmp_path / "result.pt")
    mp.spawn(
        _fsdp2_worker,
        args=(2, str(tmp_path / "store"), opt_name, result_path, info["cache_dir"]),
        nprocs=2, join=True,
    )
    _assert_close(info["params"], torch.load(result_path, weights_only=True), f"FSDP2/{opt_name}")


@pytest.mark.parametrize("opt_name", _SPLIT_OPTS)
def test_fsdp_split(opt_name, tmp_path):
    _run_fsdp_test(opt_name, tmp_path, _make_split_model, _make_split_data, "FSDP-split")


@pytest.mark.parametrize("opt_name", _INTEGRATION_OPTS)
def test_fsdp_integration(opt_name, tmp_path):
    _run_fsdp_test(opt_name, tmp_path, _make_integration_model, _make_integration_data, "FSDP-integ")
