"""Regime model. Same architecture as the tabular direction model, but
trained on the latent label `y_reval`. The OOF prediction is fed back as a
meta-feature into the direction stack."""

from __future__ import annotations

from .tabular import TabularModel, train_tabular


class RegimeModel(TabularModel):
    """Alias for clarity. The label semantics differ but the model is the
    same shape, so we lean on the tabular trainer."""


def train_regime(X, y, *, valid_X=None, valid_y=None, params=None) -> RegimeModel:
    base = train_tabular(X, y, valid_X=valid_X, valid_y=valid_y, params=params)
    return RegimeModel(feature_cols=base.feature_cols, booster=base.booster, params=base.params)
