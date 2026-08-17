"""Core CAV (Concept Activation Vector) training + TCAV directional-derivative score.

Ported from `analysis/tcav/utils.py::TCAV` (this repo, original life2vec —
not an external/foreign codebase, see module README). Only the framework-
independent math is kept: bootstrap-ensembled linear separation of
concept-vs-random activations, and the sign-agreement score between that
direction and a prediction-sensitivity gradient. The captum/dataloader
scaffolding in `analysis/tcav/utils.py` was written for the original
`TransformerEncoder.forward_with_embeddings` API and does not apply to
online2's `EventPoolingDiagnosisModelV03` — see `event_gradients.py` for the
online2-specific activation/gradient extraction that feeds these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ConceptActivationVectors:
    """Bootstrap ensemble of CAVs for one concept at one representation position.

    `coefs` has shape (n_bootstraps, D). A single CAV run gives noisy
    directions on small samples; TCAV scores are always computed against the
    whole ensemble so downstream code can report a score distribution, not a
    single point estimate — this mirrors the original module's design and is
    required, not optional (a lone bootstrap is not a concept direction).
    """

    concept_name: str
    position: str
    coefs: np.ndarray
    n_concept: int
    n_random: int
    seed: int

    def __post_init__(self) -> None:
        if self.coefs.ndim != 2:
            raise ValueError("coefs must be (n_bootstraps, D)")
        if self.n_concept < 2 or self.n_random < 2:
            raise ValueError(
                f"CAV needs >=2 concept and >=2 random examples "
                f"(got n_concept={self.n_concept}, n_random={self.n_random}) — "
                "a CAV fit on fewer is not a reliable direction"
            )


def train_cavs(
    concept_activations: np.ndarray,
    random_activations: np.ndarray,
    *,
    concept_name: str,
    position: str,
    linear_cls: Optional[BaseEstimator] = None,
    n_bootstraps: int = 200,
    n_jobs: int = 1,
    seed: int = 2021,
) -> ConceptActivationVectors:
    """Train a bootstrap ensemble of linear CAVs separating concept vs random activations.

    Same algorithm as `TCAV.calculate_cavs`/`get_training_data`: label concept
    examples 1, random examples 0, fit a bagged linear classifier, and take
    each bootstrap member's `coef_` as one CAV.
    """
    concept_activations = np.asarray(concept_activations, dtype=np.float64)
    random_activations = np.asarray(random_activations, dtype=np.float64)
    if concept_activations.shape[1:] != random_activations.shape[1:]:
        raise ValueError("concept and random activations must share feature dim")

    n_concept = concept_activations.shape[0]
    n_random = random_activations.shape[0]
    x = np.concatenate([concept_activations, random_activations], axis=0)
    y = np.concatenate([np.ones(n_concept), np.zeros(n_random)])

    base = linear_cls if linear_cls is not None else LogisticRegression(max_iter=1000)
    bagged = BaggingClassifier(
        clone(base), n_estimators=n_bootstraps, bootstrap=True, n_jobs=n_jobs, random_state=seed
    )
    bagged.fit(x, y)
    coefs = np.vstack([est.coef_ for est in bagged.estimators_])
    return ConceptActivationVectors(
        concept_name=concept_name,
        position=position,
        coefs=coefs,
        n_concept=n_concept,
        n_random=n_random,
        seed=seed,
    )


@dataclass(frozen=True)
class TCAVScore:
    """TCAV directional-derivative score distribution over the CAV bootstrap ensemble.

    `sign_fraction`: for each bootstrap CAV, the fraction of examples where the
    prediction-sensitivity gradient points in the same direction as the CAV
    (the standard TCAV score). `magnitude_fraction`: same sign test but
    weighted by dot-product magnitude, matching `TCAV.calculate_score(magnitude=True)`.
    A score near 0.5 means the concept has no consistent directional effect on
    the objective at this position — do not report a single mean without the
    per-bootstrap spread, per `E3_INTERVENTION` discipline (activation
    correlation alone is not a causal claim; this score is `E2_PREDICTIVE` at
    best until validated by ablation/steering).
    """

    concept_name: str
    position: str
    target: str
    sign_fraction: np.ndarray
    magnitude_fraction: np.ndarray
    n_examples: int

    @property
    def sign_mean(self) -> float:
        return float(np.mean(self.sign_fraction))

    @property
    def sign_std(self) -> float:
        return float(np.std(self.sign_fraction))


def tcav_score(
    gradients: np.ndarray,
    cavs: ConceptActivationVectors,
    *,
    target: str = "abnormal_risk",
) -> TCAVScore:
    """Score how often `gradients` align with each bootstrap CAV direction.

    `gradients`: (N, D) prediction-sensitivity gradients (e.g. d(abnormal risk
    margin)/dz per event, from `event_gradients.event_activation_and_gradient`).
    Same math as `TCAV.calculate_score`.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] != cavs.coefs.shape[1]:
        raise ValueError("gradients must be (N, D) with D matching the CAV feature dim")

    dots = np.einsum("nd,bd->nb", gradients, cavs.coefs)  # (N, n_bootstraps)
    sign_fraction = np.mean(dots > 0, axis=0)
    abs_dots = np.abs(dots)
    magnitude_fraction = np.sum(np.abs(dots * (dots > 0)), axis=0) / (
        np.sum(abs_dots, axis=0) + 1e-12
    )
    return TCAVScore(
        concept_name=cavs.concept_name,
        position=cavs.position,
        target=target,
        sign_fraction=sign_fraction,
        magnitude_fraction=magnitude_fraction,
        n_examples=int(gradients.shape[0]),
    )
