"""Finetune package version (independent of V2 corpus / vocab versions)."""

FINETUNE_VERSION = "0.2.0"
FINETUNE_CODENAME = "semantic_ensemble_gemma"
COMPATIBLE_V01_MODULES = (
    "src.online2.v2.event_pooling_finetune",
    "src.online2.v2.diagnosis_dataset",
)

# Artifact root — never write into v0.1 `v2_finetune/` from this package.
DEFAULT_OUTPUT_ROOT = "outputs/online2/v2_finetune_v02"
