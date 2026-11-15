import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer
import os
import hashlib
import json
from pathlib import Path


def _load_embedding_model(model_path: str):
    gpu_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, detected {gpu_count} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    print(f"Loading embedding model with vLLM from {model_path}...")
    from vllm import LLM

    tensor_parallel_size = min(gpu_count, 4)
    print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")

    vllm_kwargs = {
        "model": model_path,
        "task": "embed",
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "swap_space": 4,
        "enforce_eager": False,
    }

    vllm_model = LLM(**vllm_kwargs)
    print("vLLM model loaded successfully")
    return vllm_model, tokenizer, None


def _encode_texts_with_vllm(vllm_model, tokenizer, texts: List[str], batch_size: int = 32, logger=None):
    try:
        if logger:
            progress = logger.create_progress("Encoding texts with vLLM", len(texts))
            with progress:
                task = progress.add_task("Encoding texts", total=len(texts))

                all_embeddings = []
                num_batches = (len(texts) + batch_size - 1) // batch_size

                for batch_idx in range(0, len(texts), batch_size):
                    batch_end = min(batch_idx + batch_size, len(texts))
                    batch_texts = texts[batch_idx:batch_end]

                    outputs = vllm_model.embed(batch_texts)

                    batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                    batch_embeddings = batch_embeddings.cpu().float().numpy()

                    all_embeddings.append(batch_embeddings)
                    progress.update(task, advance=len(batch_texts))

                embeddings = np.vstack(all_embeddings)
        else:
            print(f"Encoding {len(texts)} texts with vLLM...")
            outputs = vllm_model.embed(texts)

            embeddings = torch.tensor([o.outputs.embedding for o in outputs])
            embeddings = embeddings.cpu().float().numpy()

        return embeddings

    except Exception as e:
        print(f"Error using vLLM for embeddings: {e}")
        print("Note: Make sure vLLM is loaded with task='embed' and vllm>=0.8.5")
        raise


def encode_texts_with_embedding_model(model_path: str, texts: List[str], batch_size: int = 32, logger=None,
                           model=None, tokenizer=None, device=None):
    if model is not None and tokenizer is not None:
        return _encode_texts_with_vllm(model, tokenizer, texts, batch_size, logger)

    model, tokenizer, _ = _load_embedding_model(model_path)

    try:
        embeddings = _encode_texts_with_vllm(model, tokenizer, texts, batch_size, logger)
    finally:
        del model
        torch.cuda.empty_cache()

    return embeddings


def _generate_cache_key(args, train_texts: List[str], val_texts: List[str], test_texts: List[str]) -> str:
    config = {
        'dataset': getattr(args, 'dataset', 'unknown'),
        'embedding_model_name': getattr(args, 'embedding_model_name', 'unknown'),
    }

    text_hashes = []
    for texts in [train_texts, val_texts, test_texts]:
        if texts:
            text_hashes.append(f"len:{len(texts)}")
            text_content_hashes = []
            for text in texts:
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                text_content_hashes.append(text_hash)
            combined_text_hash = hashlib.md5(''.join(text_content_hashes).encode('utf-8')).hexdigest()
            text_hashes.append(combined_text_hash)
        else:
            text_hashes.append("empty")

    cache_data = json.dumps(config, sort_keys=True) + ''.join(text_hashes)

    cache_key = hashlib.md5(cache_data.encode('utf-8')).hexdigest()
    return cache_key


def _get_cache_path(args, cache_key: str) -> Path:
    dataset = getattr(args, 'dataset', 'unknown')
    method = getattr(args, 'method', 'unknown')
    model_name = getattr(args, 'embedding_model_name', 'unknown')

    cache_dir = Path('cache') / 'embeddings' / dataset / method / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f'embeddings_{cache_key}.pt'
    return cache_file


def _save_embeddings(cache_path: Path, train_embeddings: np.ndarray,
                     val_embeddings: np.ndarray, test_embeddings: np.ndarray):
    cache_data = {
        'train_embeddings': train_embeddings,
        'val_embeddings': val_embeddings,
        'test_embeddings': test_embeddings,
    }
    torch.save(cache_data, cache_path)
    print(f"Embeddings saved to cache: {cache_path}")


def _load_embeddings(cache_path: Path) -> tuple:
    try:
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load cache file {cache_path}: {e}. The file may be corrupted.")

    if not isinstance(cache_data, dict):
        raise ValueError(f"Cache file {cache_path} does not contain a dictionary. Got type: {type(cache_data)}")

    required_keys = ['train_embeddings', 'val_embeddings', 'test_embeddings']
    missing_keys = [key for key in required_keys if key not in cache_data]
    if missing_keys:
        available_keys = list(cache_data.keys())
        raise KeyError(f"Cache file {cache_path} is missing required keys: {missing_keys}. "
                      f"Available keys: {available_keys}. The cache file may be from an older version.")

    train_embeddings = cache_data['train_embeddings']
    val_embeddings = cache_data['val_embeddings']
    test_embeddings = cache_data['test_embeddings']

    if isinstance(train_embeddings, torch.Tensor):
        train_embeddings = train_embeddings.cpu().numpy()
    if isinstance(val_embeddings, torch.Tensor):
        val_embeddings = val_embeddings.cpu().numpy()
    if isinstance(test_embeddings, torch.Tensor):
        test_embeddings = test_embeddings.cpu().numpy()

    if not isinstance(train_embeddings, np.ndarray):
        train_embeddings = np.array(train_embeddings)
    if not isinstance(val_embeddings, np.ndarray):
        val_embeddings = np.array(val_embeddings)
    if not isinstance(test_embeddings, np.ndarray):
        test_embeddings = np.array(test_embeddings)

    print(f"Embeddings loaded from cache: {cache_path}")
    return train_embeddings, val_embeddings, test_embeddings


def calculate_semantic_embedding(args, train_dataset, val_dataset, test_dataset, logger=None):
    if args.embedding_model_name == 'qwen3-embedding-8b':
        model_path = args.embedding_model_path
        if model_path is None or not os.path.exists(model_path):
            raise ValueError(f"Embedding model path not found: {model_path}")

        train_texts = train_dataset.get('detail', [])
        val_texts = val_dataset.get('detail', [])
        test_texts = test_dataset.get('detail', [])

        if not train_texts and not val_texts and not test_texts:
            raise ValueError("No 'detail' field found in datasets. Please ensure the datasets have been processed with prompt wrapper.")

        cache_key = _generate_cache_key(args, train_texts, val_texts, test_texts)
        cache_path = _get_cache_path(args, cache_key)

        print(f"[Cache Debug] Current cache key: {cache_key}")
        print(f"[Cache Debug] Path: {cache_path}, Exists: {cache_path.exists()}")
        if not cache_path.exists() and cache_path.parent.exists():
            existing = list(cache_path.parent.glob('embeddings_*.pt'))
            print(f"[Cache Debug] Found {len(existing)} existing cache files in directory")
            existing_keys = [f.stem.replace('embeddings_', '') for f in existing]
            print(f"[Cache Debug] Existing cache keys: {existing_keys[:3]}..." if len(existing_keys) > 3 else f"[Cache Debug] Existing cache keys: {existing_keys}")

        if cache_path.exists():
            if logger:
                logger.info(f"Found cached embeddings, loading from {cache_path}")
            else:
                print(f"Found cached embeddings, loading from {cache_path}")
            try:
                train_embeddings, val_embeddings, test_embeddings = _load_embeddings(cache_path)
                train_dataset['semantic_embedding'] = torch.from_numpy(train_embeddings).float()
                val_dataset['semantic_embedding'] = torch.from_numpy(val_embeddings).float()
                test_dataset['semantic_embedding'] = torch.from_numpy(test_embeddings).float()

                print(f"Train embeddings shape: {train_embeddings.shape}")
                print(f"Val embeddings shape: {val_embeddings.shape}")
                print(f"Test embeddings shape: {test_embeddings.shape}")

                return train_dataset, val_dataset, test_dataset
            except Exception as e:
                print(f"[Cache Debug] Load failed: {type(e).__name__}: {e}")
                if logger:
                    logger.warning(f"Failed to load cache: {e}. Recomputing embeddings...")
                else:
                    print(f"Failed to load cache: {e}. Recomputing embeddings...")

        if logger:
            logger.info("Computing embeddings (cache miss or invalid cache)...")
        else:
            print("Computing embeddings (cache miss or invalid cache)...")

        batch_size = getattr(args, 'vllm_batch_size', 4)

        model, tokenizer, _ = _load_embedding_model(model_path)

        try:
            print("Encoding train dataset with vLLM...")
            train_embeddings = _encode_texts_with_vllm(
                model, tokenizer, train_texts, batch_size=batch_size, logger=logger
            )

            print("Encoding val dataset with vLLM...")
            val_embeddings = _encode_texts_with_vllm(
                model, tokenizer, val_texts, batch_size=batch_size, logger=logger
            )

            print("Encoding test dataset with vLLM...")
            test_embeddings = _encode_texts_with_vllm(
                model, tokenizer, test_texts, batch_size=batch_size, logger=logger
            )
        finally:
            if model is not None:
                try:
                    if logger:
                        logger.info("Releasing vLLM embedding model memory...")
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    if logger:
                        logger.success("vLLM embedding model memory released successfully")
                    else:
                        print("vLLM embedding model memory released successfully")
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to release vLLM embedding model memory: {e}")
                    else:
                        print(f"Warning: Failed to release vLLM embedding model memory: {e}")

        try:
            _save_embeddings(cache_path, train_embeddings, val_embeddings, test_embeddings)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save embeddings to cache: {e}")
            else:
                print(f"Warning: Failed to save embeddings to cache: {e}")

        train_dataset['semantic_embedding'] = torch.from_numpy(train_embeddings).float()
        val_dataset['semantic_embedding'] = torch.from_numpy(val_embeddings).float()
        test_dataset['semantic_embedding'] = torch.from_numpy(test_embeddings).float()

        print(f"Train embeddings shape: {train_embeddings.shape}")
        print(f"Val embeddings shape: {val_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")

    else:
        raise ValueError(f"Unsupported embedding model: {args.embedding_model_name}")

    return train_dataset, val_dataset, test_dataset
