from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "v1.0"
OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "outputs" / "berry2vec" / VERSION

PRETRAIN_CKPT = (
    PROJECT_ROOT / "checkpoints/agri/pre_training/1.0/1.0-epoch=29.ckpt"
)
FINETUNE_CKPT = (
    PROJECT_ROOT / "checkpoints/agri/fruiting/l2v/1.0/1.0-epoch=06.ckpt"
)
VOCAB_PATH = PROJECT_ROOT / "data/processed/vocab/agri_pretrain_set/result.tsv"
POPULATION_PATH = (
    PROJECT_ROOT / "data/processed/populations/agri_fruiting_set/population/result.pkl"
)

TOKEN_EMBEDDINGS_TSV = OUTPUT_ROOT / "token_embeddings.tsv"
PLANT_REPR_NPY = OUTPUT_ROOT / "plant_repr.npy"
META_PKL = OUTPUT_ROOT / "meta.pkl"
SALIENCY_PKL = OUTPUT_ROOT / "saliency.pkl"
FIGURES_DIR = OUTPUT_ROOT / "figures"
