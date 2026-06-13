"""Tests for the env_io and training service seams (incl. review-found fixes)."""

from __future__ import annotations

import os

from magpy.services import apply_to_environ, read_env, write_env
from magpy.services import training as T


def test_env_roundtrip_omits_empty(tmp_path):
    p = tmp_path / "magpy.env"
    write_env(p, {"XC_API_KEY": "abc", "EBIRD_API_KEY": "", "HF_TOKEN": "tok val"})
    assert read_env(p) == {"XC_API_KEY": "abc", "HF_TOKEN": "tok val"}


def test_apply_to_environ_sets_and_unsets(monkeypatch):
    # Clearing a key must remove it from the live environment, not leave it stale.
    monkeypatch.setenv("XC_API_KEY", "oldkey")
    apply_to_environ({"XC_API_KEY": "", "EBIRD_API_KEY": "newkey"})
    assert "XC_API_KEY" not in os.environ
    assert os.environ["EBIRD_API_KEY"] == "newkey"


def test_run_training_filters_unknown_kwargs(monkeypatch):
    seen = {}

    def fake_train_ast(**kw):
        seen.clear()
        seen.update(kw)
        return type("R", (), {"model_path": "m", "epochs": 1,
                              "final_accuracy": None, "final_loss": None})()

    monkeypatch.setattr(T, "train_ast", fake_train_ast)
    T.run_training("ds", "dir", {"num_train_epochs": 3, "bogus": 1})
    assert "bogus" not in seen
    assert seen["num_train_epochs"] == 3
    assert seen["train_dataset"] == "ds" and seen["training_dir"] == "dir"


def test_run_training_disables_load_best_for_incompatible_strategies(monkeypatch):
    seen = {}

    def fake_train_ast(**kw):
        seen.clear()
        seen.update(kw)
        return type("R", (), {"model_path": "m", "epochs": 1,
                              "final_accuracy": None, "final_loss": None})()

    monkeypatch.setattr(T, "train_ast", fake_train_ast)
    # Compatible: leave train_ast's default (True) alone.
    T.run_training("ds", "dir", {"eval_strategy": "epoch", "save_strategy": "epoch"})
    assert "load_best_model_at_end" not in seen
    # Incompatible: force False so TrainingArguments won't raise after the download.
    T.run_training("ds", "dir", {"eval_strategy": "no", "save_strategy": "epoch"})
    assert seen["load_best_model_at_end"] is False
