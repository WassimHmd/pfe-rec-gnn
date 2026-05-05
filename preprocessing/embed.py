# src/embed.py
import gzip
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-base-en-v1.5"   # 768 dim, ~109M params
BATCH_SIZE  = 256                         # réduis à 128 si OOM sur GPU
MAX_CHARS   = 2000                        # ~512 tokens

BOOKS_INPUT   = "data/raw/goodreads_books_poetry.json.gz"
REVIEWS_INPUT = "data/raw/goodreads_reviews_poetry.json.gz"
BOOKS_OUTPUT  = "data/embeddings/books_bge_base_768.h5"
REVIEWS_OUTPUT= "data/embeddings/reviews_bge_base_768.h5"
# ────────────────────────────────────────────────────────────────────────


def load_model():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle sur : {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Dimension de sortie : {model.get_sentence_embedding_dimension()}")
    return model


def embed_books(model, input_path: str, output_path: str):
    """Encode title + description pour chaque livre."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n── Books : lecture de {input_path}")
    ids, texts = [], []

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            book_id = r.get("book_id", "")
            title   = r.get("title", "") or ""
            desc    = r.get("description", "") or ""

            # Format spec §4.2 : "title. description"
            text = f"{title}. {desc}".strip()[:MAX_CHARS]

            ids.append(book_id)
            texts.append(text if text else "[no text]")

    print(f"  {len(ids)} livres trouvés")

    print("  Encodage en cours...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity directe après
        convert_to_numpy=True,
    )

    print(f"  Sauvegarde → {output_path}")
    with h5py.File(output_path, "w") as hf:
        # IDs sous forme de bytes (compatibilité HDF5)
        hf.create_dataset("book_id",   data=np.array(ids, dtype="S30"))
        hf.create_dataset("embedding", data=embeddings, dtype="float32",
                          compression="gzip", compression_opts=4)

    print(f"  ✓ {len(ids)} embeddings de dim {embeddings.shape[1]} sauvegardés")


def embed_reviews(model, input_path: str, output_path: str):
    """Encode review_text pour chaque review non vide."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n── Reviews : lecture de {input_path}")
    ids, texts = [], []
    skipped = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("review_text", "") or "").strip()

            if not text:          # spec §7.1 : drop empty reviews
                skipped += 1
                continue

            ids.append(r.get("review_id", ""))
            texts.append(text[:MAX_CHARS])

    print(f"  {len(ids)} reviews valides ({skipped} vides ignorées)")

    print("  Encodage en cours...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"  Sauvegarde → {output_path}")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("review_id", data=np.array(ids, dtype="S40"))
        hf.create_dataset("embedding", data=embeddings, dtype="float32",
                          compression="gzip", compression_opts=4)

    print(f"  ✓ {len(ids)} embeddings de dim {embeddings.shape[1]} sauvegardés")


def verify(output_path: str, id_key: str):
    """Vérifie rapidement le fichier produit."""
    with h5py.File(output_path, "r") as hf:
        ids  = hf[id_key][:]
        embs = hf["embedding"][:]
        print(f"\n  Vérification {output_path}")
        print(f"  IDs     : {len(ids)}")
        print(f"  Shape   : {embs.shape}")     # (N, 768)
        print(f"  dtype   : {embs.dtype}")     # float32
        # norme ≈ 1.0 si normalize_embeddings=True
        norms = np.linalg.norm(embs[:5], axis=1)
        print(f"  Normes (5 premiers) : {norms.round(4)}")


if __name__ == "__main__":
    model = load_model()

    embed_books(model,   BOOKS_INPUT,   BOOKS_OUTPUT)
    embed_reviews(model, REVIEWS_INPUT, REVIEWS_OUTPUT)

    verify(BOOKS_OUTPUT,   "book_id")
    verify(REVIEWS_OUTPUT, "review_id")

    print("\n✅ Terminé.")