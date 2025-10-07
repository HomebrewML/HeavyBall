#!/usr/bin/env bash
set -euo pipefail

# Reduce noisy plugin autoloading and suppress known informational warnings.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:pkg_resources is deprecated as an API:UserWarning,ignore:CUDA initialization:UserWarning,ignore:Can't initialize NVML:UserWarning"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

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
test/test_psgd_precond_init_stability.py::test_stable_exp_scalar -k dtype0
test/test_psgd_precond_init_stability.py::test_stable_exp_tensor -k dtype0
test/test_psgd_precond_init_stability.py::test_lse_mean -k dtype0
test/test_psgd_precond_init_stability.py::test_stable_exp_scalar -k dtype2
test/test_psgd_precond_init_stability.py::test_stable_exp_tensor -k dtype2
test/test_psgd_precond_init_stability.py::test_mean_root[dtype0-4-16]
test/test_psgd_precond_init_stability.py::test_mean_root[dtype0-10-512]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype0-3-5-16]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype0-9-4-64]
test/test_psgd_precond_init_stability.py::test_lse_mean -k dtype2
test/test_psgd_precond_init_stability.py::test_mean_root[dtype2-4-16]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype2-9-4-128]
test/test_psgd_precond_init_stability.py::test_divided_root[dtype2-15-15-512]
EOF
fi
