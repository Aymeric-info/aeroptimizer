"""Contract tests for `config` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

from pathlib import Path

import pytest

MBB = Path(__file__).resolve().parents[2] / "configs" / "mbb_2d.yaml"


def test_unknown_key_rejected(tmp_path):
    r"""C-CFG-01 — La configuration est un fichier YAML validé par des `dataclass` typées ; toute clé inconnue est une erreur (pa…"""
    from aeroptimizer.config import load_config
    from aeroptimizer.errors import ConfigError

    p = tmp_path / "bad.yaml"
    p.write_text(MBB.read_text() + "typo_key: 1\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_config(p)


def test_resolved_roundtrip(tmp_path):
    r"""C-CFG-02 — Un même YAML, graine et device ⇒ même `config.resolved.yaml` (les valeurs par défaut sont explicitées)."""
    import yaml

    from aeroptimizer.config import load_config, resolved_yaml

    cfg = load_config(MBB)
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(resolved_yaml(cfg))
    assert load_config(resolved) == cfg
    assert yaml.safe_load(resolved.read_text())["optim"]["max_iter"] == 200  # defaults explicit


def test_calibration_params_explicit(tmp_path):
    r"""C-CFG-03 — Les paramètres marqués ⚠ dans FORMALISM (`c_b`, `k_w`, `E_z`, …) n'ont **pas** de valeur par défaut silencieus…"""
    from aeroptimizer.config import load_config
    from aeroptimizer.errors import ConfigError

    drone = MBB.parent / "drone_arm_2d.yaml"
    assert load_config(drone).flow.c_b == 0.3  # shipped example provides them (marked TODO)
    text = drone.read_text()
    for key in ("k_w", "c_b"):
        p = tmp_path / f"no_{key}.yaml"
        p.write_text("".join(ln + "\n" for ln in text.splitlines() if not ln.strip().startswith(key + ":")))
        with pytest.raises(ConfigError, match=key):
            load_config(p)
