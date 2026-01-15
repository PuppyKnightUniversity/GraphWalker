from typing import List
import os
import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _compute_qwen3_embeddings(model_path: str, texts: List[str], batch_size: int = 8) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype="auto")
    mdl.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            inp = tok(batch, padding=True, truncation=True, return_tensors="pt")
            inp = {k: v.to(mdl.device) for k, v in inp.items()}
            o = mdl(**inp)
            h = o.last_hidden_state
            m = inp["attention_mask"].unsqueeze(-1)
            s = (h * m).sum(dim=1)
            l = m.sum(dim=1)
            mean = s / (l + 1e-12)
            out.append(mean.cpu().numpy())
    embs = np.concatenate(out, axis=0)
    del mdl, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _normalize_rows(embs)


def _compute_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=4096)
    mat = vec.fit_transform(texts)
    return _normalize_rows(mat.toarray())


def get_or_build_embeddings(args, logger, texts: List[str], cache_dir: str, cache_name: str, batch_size: int = 8) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_name}.npy")
    if os.path.exists(path):
        if logger:
            logger.info(f"Loading cached embeddings from {path}")
        return np.load(path)
    if logger:
        logger.processing_start(f"Computing embeddings: {cache_name}")
    name = getattr(args, "embedding_model_name", "smart")
    model_path = None
    try:
        from args.ehrbase_args import LLM_PATH_MAP
        model_path = LLM_PATH_MAP.get(name)
    except Exception:
        model_path = None
    try:
        if name == "qwen3-embedding-8b" and model_path:
            embs = _compute_qwen3_embeddings(model_path, texts, batch_size=batch_size)
        else:
            embs = _compute_tfidf_embeddings(texts)
    except Exception:
        embs = _compute_tfidf_embeddings(texts)
    np.save(path, embs)
    if logger:
        logger.processing_complete(f"Computing embeddings: {cache_name}")
    return embs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T