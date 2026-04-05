from pathlib import Path

from anima_lsfdnet.config import load_config


def test_load_default_config() -> None:
    cfg = load_config("configs/default.toml")
    assert cfg.train.lr == 1e-4
    assert cfg.model.patch_size == 8


def test_fallback_merge(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text("[train]\nbatch_size=4\n")
    cfg = load_config(override, "configs/default.toml")
    assert cfg.train.batch_size == 4
    assert cfg.train.lr == 1e-4
