from types import SimpleNamespace

from engine_IC import _resolve_max_epochs


def test_resolve_max_epochs_uses_explicit_cfg_value():
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(MAX_EPOCH=1))

    assert _resolve_max_epochs(cfg) == 1


def test_resolve_max_epochs_keeps_legacy_default_for_generic_cfg_default():
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(MAX_EPOCH=400))

    assert _resolve_max_epochs(cfg) == 10
