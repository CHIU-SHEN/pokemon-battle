#!/bin/bash

activate_pokemon_env() {
  # Some cluster bashrc files reference unset variables. Keep nounset disabled
  # while loading shell startup files and conda hooks.
  set +u

  if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" || true
  fi

  if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/share/apps/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found; please initialize conda or update jobs/slurm_env.sh" >&2
    exit 1
  fi

  conda activate pokemon-tcg
}

