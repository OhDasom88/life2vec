# Finetune v0.3 script root

Canonical plan: [`docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md`](../../../docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md)  
Follow-up build plan: [`docs/online2/DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md`](../../../docs/online2/DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md)

| | path |
|--|------|
| Code | `src/online2/v2/finetune_v03/` |
| Scripts | `scripts/online2_v2/v03/` |
| Artifacts | `outputs/online2/v2_finetune_v03/` |
| Sweeps | `conf/sweep/finetune_v03_s{1,2,3}_*.yaml` (shared group `finetune_v03_improvement_r1`) |

## Train modes

```bash
# Repeated group-stratified 3-fold × 3 seeds (tuning)
python scripts/online2_v2/v03/run_diagnosis_finetune_v03.py --mode cv_repeated --wandb \
  --wandb-group finetune_v03_improvement_r1 --wandb-job-type sweep_1_loss_binary \
  --monitor val_loss --pos-weight auto

# Final 5-fold after HP freeze
python scripts/online2_v2/v03/run_diagnosis_finetune_v03.py --mode cv_final5 --n-folds 5
```

Defaults (PLAN §19): `monitor=val_loss`, `pos_weight=auto`, `lambda_consistency=0.05`,
`lambda_prototype=0.05`, `lambda_supcon=0`, task-specific query + attention residual ON,
image adapter ON (not swept), reject/open-set ON.

## Three-stage sweep (same W&B group)

```bash
export WANDB_GROUP=finetune_v03_improvement_r1
wandb sweep --project Berry2Vec --entity dasom-oh conf/sweep/finetune_v03_s1_loss_binary.yaml
wandb agent dasom-oh/Berry2Vec/<SWEEP_ID> --count 24
# then s2, s3 with same WANDB_GROUP — filter group in UI
```

`pos_weight` absolute values or `auto`; binary thresholds logged at `{0.25,0.5,0.75}`.
