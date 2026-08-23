# Pinned upstream oracles

Each directory next to this file is a shallow clone of the upstream open-source
implementation the Matrix uses as its accuracy oracle and speed baseline. The
Matrix runs this code as-is; it never rewrites its math. These clones are dev-time
dependencies of `benchmarks/matrix.py` only: nothing here is imported by the
`heavyball` package, and nothing from here ships in a release.

| directory      | upstream                             | pin     | license                                     |
| -------------- | ------------------------------------ | ------- | ------------------------------------------- |
| `Muon`         | https://github.com/KellerJordan/Muon         | f98f1ca | MIT                                         |
| `SOAP`         | https://github.com/nikhilvyas/SOAP           | a1e5535 | MIT                                         |
| `kron_torch`   | https://github.com/evanatyourservice/kron_torch | 884427c | CC-BY-4.0 (attribution)                     |
| `schedule_free`| https://github.com/facebookresearch/schedule_free | 70785b5 | Apache-2.0                                  |
| `psgd_torch`   | https://github.com/lixilinx/psgd_torch       | c86b1cb | none stated; dev-time oracle use only, never copied |

`torch.optim.AdamW` and `torch.optim.SGD` complete the baseline set; PyTorch's
own license terms cover them.

`kron_torch` is the direct lineage of our PSGD Kron (the `PSGDKron` docstring
cites it at a pinned commit); `psgd_torch` is Xi-Lin Li's original PSGD. QSGD and
PSGD-LRA have no same-semantics upstream with a gradient-replay API (the original
LRA is closure/HVP-driven), so their cells carry `accuracy_baseline: null` and
stay on the fp64 self-oracle margin gate.

## Second wave (2026-08-20)

| directory | upstream | pin | license |
| --------- | -------- | --- | ------- |
| `adopt` | https://github.com/iShohei220/adopt | 6468572 | Apache-2.0 |
| `ademamix-optimizer-pytorch` | https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch | 0f52410 | MIT (community port; AdEMAMix is Apple's algorithm) |
| `mars` | https://github.com/AGI-Arena/MARS | 4831e28 | Apache-2.0 |
| `scion` | https://github.com/LIONS-EPFL/scion | f58a393 | MIT |

Plus `torch.optim.NAdam` and `torch.optim.RMSprop`. This brings the upstream-compared
compositions to 13 of 39; the rest have no same-semantics upstream and stay on the
fp64 self-oracle gate, listed in `baselineless.jsonl`.

## Third wave (2026-08-20)

| directory | upstream | pin | license |
| --------- | -------- | --- | ------- |
| `laprop-optimizer` | https://github.com/Z-T-WANG/LaProp-Optimizer | a419916 | MIT |
| `c-optim` | https://github.com/kyleliang919/C-Optim | a506ee3 | MIT |
| `kl-methods` | https://github.com/yorkerlin/KL-Methods | a02e622 | none stated; dev-time oracle use only |

LaProp replays cleanly. C-Optim's AdamW step is a torch.compile(fullgraph=True)
generator and crashes eager replay on torch 2.12; yorkerlin's KLOpt calls .mul_
on a list state -- both recorded per-row as upstream failures, not hidden.
MuonWithAuxAdam (already cloned in Muon/) remains unwired: it needs param-group
construction our flat-params replay does not express.

## Fourth wave (2026-08-20)

| directory | upstream | pin | license |
| --------- | -------- | --- | ------- |
| `orthograd` | https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability | 720d244 | MIT |
| `pytorch_optimizer` | https://github.com/kozistr/pytorch_optimizer | b452a0f | Apache-2.0 |

OrthoGrad wraps official LaProp as its base (wrap_base machinery) for
ortho_laprop. kozistr's AdamC covers adamc. MuonWithAuxAdam (already cloned)
is wired via param-group construction for muon_adamw, but its step issues
collectives: single-process replays record the PG error per row -- it is a
distributed-cell-only upstream. Twenty-two of 39 compositions now carry an
upstream; the remaining seventeen are our own algorithms (heavy_* family,
hyperball, soap/muon/laprop variant compositions, LATHER, SUDS, MSAM,
unscaled_adam, sign/ortho-order compositions) with no upstream anywhere.

## Fifth wave (2026-08-20)

| directory | upstream | pin | license |
| --------- | -------- | --- | ------- |
| `msam` | https://github.com/MarlonBecker/MSAM | 780fa4f | MIT |

Official Momentum-SAM (NeurIPS 2025) covers msam. Twenty-three of 39 compared.

The remaining sixteen, with provenance checked against each facade's docstring:
heavy_* family, hyperball, unscaled_adam, LATHER, SUDS cite only
HomebrewML/HeavyBall itself (our own algorithms); soap_nadam, soap_ademamix,
soap_laprop, muon_laprop, laprop_ortho, sign_laprop are our own compositions
of published parts with no combined official form.
