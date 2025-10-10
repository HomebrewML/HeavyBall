#!/usr/bin/env bash
set -euo pipefail

# Reduce noisy plugin autoloading and suppress known informational warnings.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:pkg_resources is deprecated as an API:UserWarning,ignore:CUDA initialization:UserWarning,ignore:Can't initialize NVML:UserWarning"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

PYTEST_FLAGS=(--maxfail=1 --disable-warnings -q --color=no --code-highlight=no)

run_pytest() {
  echo "::group::pytest $*"
  pytest "${PYTEST_FLAGS[@]}" "$@"
  echo "::endgroup::"
}

run_list() {
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    read -r -a args <<<"${line}"
    run_pytest "${args[@]}"
  done
}

run_list <<'EOF'
test/test_toy_training.py
test/test_migrate_cli.py
test/test_cpu_features.py
test/test_chainable_cpu.py
test/test_helpers_cpu.py
test/test_utils_cpu.py
test/test_stochastic_utils_cpu.py
test/test_optimizer_cpu_smoke.py
test/test_psgd_precond_init_stability.py::test_stable_exp_scalar -k dtype1
test/test_psgd_precond_init_stability.py::test_stable_exp_tensor -k dtype1
test/test_psgd_precond_init_stability.py::test_lse_mean -k dtype1
test/test_psgd_precond_init_stability.py::test_mean_root[dtype1-4-16]
test/test_psgd_precond_init_stability.py::test_mean_root[dtype2-10-512]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype1-3-5-16]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype2-9-4-64]
EOF

if [[ ${1:-} == push ]]; then
  run_list <<'EOF'
test/test_toy_training.py
test/test_migrate_cli.py
EOF
fi
