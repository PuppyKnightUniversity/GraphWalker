"""GraphWalker: cosine k-NN, Leiden cohorts and greedy frontier search."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
import torch
import numpy as np
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logger import get_logger
import igraph as ig
import leidenalg

# Global caches for model and tokenizer to avoid reloading
_metric_model_cache = {}
_metric_tokenizer_cache = {}

def select_graph_walker_examples(args, test_dataset, train_dataset, num_examples: int, logger=None) -> List[List[Dict[str, Any]]]:
    '''
    Select graph walker examples for all test patients
    Args:
        args: the arguments
        test_dataset: the test dataset
        train_dataset: the train dataset
        num_examples: the number of examples to select
    Returns:
        the selected examples for all test patients
    '''
    # Initialize logger if not provided
    if logger is None:
        logger = get_logger("GraphWalker")

    if getattr(args, 'is_api', False):
        raise ValueError('GraphWalker scoring requires local token log-probabilities')
    if num_examples < 0:
        raise ValueError('num_examples must be non-negative')
    if num_examples == 0 or not train_dataset['detail']:
        return [[] for _ in test_dataset['detail']]

    # Get embedding key based on embedding model name
    if args.embedding_model_name == 'smart':
        emd_key = 'smart_embedding'
    elif args.embedding_model_name == 'qwen3-embedding-8b':
        emd_key = 'semantic_embedding'
    else:
        raise ValueError(f"Unsupported embedding model: {args.embedding_model_name}")

    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []

    # build graph
    neighbor_num = args.graph_walker_neighbor_num
    graph = _build_graph(args, train_dataset, neighbor_num=neighbor_num, logger=logger, emd_key=emd_key)

    # find patient cohorts via Leiden and compute centroids
    cohort_assignments, cohort_to_patients, centroids, igraph_g = _leiden_cluster_patients(
        train_dataset, graph, args=args, logger=logger, emd_key=emd_key
    )

    # load vllm model
    vllm_model = _load_vllm_model(args, logger=logger)

    patient_num = len(test_dataset['detail'])
    progress = logger.create_progress("Selecting GraphWalker examples for all test patients", patient_num)
    try:
        with progress:
            task = progress.add_task("Selecting GraphWalker examples for all test patients", total=patient_num)
            for i in range(patient_num):
                patient_example = {}
                for key in test_dataset.keys():
                    patient_example[key] = test_dataset[key][i]
                ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS.append(
                    select_graph_walker_examples_for_single_patient(
                        args, patient_example, train_dataset, graph,
                        centroids, cohort_to_patients, num_examples,
                        emb_key=emd_key, mode=args.graph_walker_mode, logger=logger,
                        vllm_model=vllm_model
                    )
                )
                progress.update(task, advance=1)
    
        logger.processing_complete("Selecting GraphWalker examples for all test patients")
    
    finally:
        del vllm_model
        _release_vllm_model(logger=logger)

    return ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS

def _load_vllm_model(args, logger=None)->LLM:

    from vllm import LLM

    # detect gpu count
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError('GraphWalker entropy scoring requires a CUDA GPU')
    tensor_parallel_size = min(gpu_count, 4)  # Limit to 4 GPUs max for stability
    print(f"Detected {gpu_count} GPUs, using {tensor_parallel_size} for tensor parallelism")
    # Check GPU memory
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3  # GB
            print(f"GPU {i} ({props.name}): {total_mem:.2f} GB total memory")
    # Check if model path exists
    import os
    model_path = args.llm_local_path
    max_model_len = args.vllm_max_model_len
    gpu_memory_utilization = args.vllm_gpu_memory_utilization
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    # initialize vllm kwargs
    vllm_kwargs = {"model": model_path,
                   "tensor_parallel_size": tensor_parallel_size,
                   "trust_remote_code": True,
                   "dtype": "bfloat16",
                   "gpu_memory_utilization": gpu_memory_utilization,  # Control GPU memory usage to avoid OOM
                   "max_model_len": max_model_len,  # Maximum sequence length (can be increased if GPU memory allows)
                   "swap_space": 4, "seed": getattr(args, "seed", 3407)}
    logger.info(f"Initializing vLLM with kwargs: {vllm_kwargs}")
    vllm_model = LLM(**vllm_kwargs)
    logger.success("vLLM model initialized successfully")
    return vllm_model

def _release_vllm_model(logger=None):
    """Collect unused model objects after the caller releases its reference."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as error:
        if logger:
            logger.warning(f'GPU cache cleanup failed: {error}')


def _build_graph(args, train_dataset, neighbor_num: int = 8, logger=None,
                 emd_key: str = 'smart_embedding'):
    """Build the union-symmetrized k-NN graph used by BOTH Leiden and traversal.

    Cosine values determine neighbor ranks and are retained as metadata. Leiden
    uses binary adjacency, as in Eq. (1); negative similarities are not negative
    modularity weights. Symmetrization can give nodes more than k neighbors.
    """
    if neighbor_num < 0:
        raise ValueError("neighbor_num must be non-negative")
    embeddings = torch.as_tensor(train_dataset[emd_key], dtype=torch.float32)
    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError("Patient embeddings must be a finite 2-D array")
    n = len(embeddings)
    adjacency = {i: {} for i in range(n)}
    k = min(neighbor_num, max(0, n - 1))
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # Avoid materializing an N x N similarity matrix for the whole EHR base.
    for start in range(0, n, 1024):
        similarity = normalized[start:start + 1024] @ normalized.T
        for row, i in enumerate(range(start, min(start + 1024, n))):
            similarity[row, i] = float('-inf')
            # Stable ordering gives patient-index tie breaks for equal embeddings.
            indices = torch.argsort(similarity[row], descending=True, stable=True)[:k]
            for j in indices.tolist():
                weight = float(similarity[row, j])
                adjacency[i][j] = weight
                adjacency[j][i] = weight
    graph = {i: sorted(neighbors.items()) for i, neighbors in adjacency.items()}
    if logger:
        logger.success("Symmetrized patient graph built successfully")
    return graph

def _leiden_cluster_patients(
    train_dataset, graph: Dict[int, List[Tuple[int, float]]], args=None,
    logger=None, emd_key: str = 'smart_embedding',
    resolution_parameter: float = None, random_state: int = None
) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray, ig.Graph]:
    '''
    Perform Leiden clustering on all patients to identify patient cohorts and compute centroids
    Args:
        train_dataset: the train dataset containing patient embeddings
        graph: the graph built from train dataset, dict where graph[i] = [(neighbor_idx, weight), ...]
        args: arguments object (optional, used to get hyperparameters)
        logger: optional logger instance
        emd_key: key for embeddings in train_dataset ('smart_embedding' or 'semantic_embedding')
        resolution_parameter: resolution parameter for Leiden algorithm (higher = more clusters)
                              If None, will try to get from args.graph_walker_leiden_resolution,
                              otherwise defaults to 1.0
        random_state: random seed for reproducibility
                     If None, will try to get from args.seed, otherwise defaults to 42
    Returns:
        tuple: (cohort_assignments, cohort_to_patients, centroids, igraph_g)
        cohort_assignments: numpy array of shape (dataset_size,) where each element is the cohort ID
        cohort_to_patients: dict mapping cohort_id -> list of patient indices in that cohort
        centroids: numpy array of shape (n_clusters, embedding_dim) containing cluster centers
        igraph_g: the igraph Graph object used for clustering (for visualization)
    '''
    # Get hyperparameters from args or use defaults
    if resolution_parameter is None:
        if args is not None and hasattr(args, 'graph_walker_leiden_resolution'):
            resolution_parameter = args.graph_walker_leiden_resolution
        else:
            resolution_parameter = 1.0

    if random_state is None:
        if args is not None and hasattr(args, 'seed'):
            random_state = args.seed
        else:
            random_state = 42  # Default seed

    if logger:
        logger.info(f"Performing Leiden clustering with resolution={resolution_parameter}, seed={random_state}...")

    # Get dataset size from graph (graph contains all nodes from 0 to dataset_size-1)
    dataset_size = len(graph)

    g = ig.Graph(directed=False)
    g.add_vertices(dataset_size)

    # Leiden uses binary edges; traversal retains cosine similarities.
    edges = sorted({(min(i, j), max(i, j))
                    for i, neighbors in graph.items() for j, _ in neighbors if i != j})
    g.add_edges(edges)
    if dataset_size == 0:
        dim = train_dataset[emd_key].shape[1]
        return np.array([], dtype=int), {}, np.empty((0, dim)), g
    if not edges:
        # No modularity denominator: isolated patients form singleton cohorts.
        partition = ig.VertexClustering(g, membership=list(range(dataset_size)))
    else:
        # Degree-based configuration null model with adjustable resolution.
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution_parameter, seed=random_state,
        )

    # Get cluster assignments
    cohort_assignments = np.array(partition.membership)  # Shape: (dataset_size,)
    actual_n_clusters = len(partition)

    if logger:
        logger.info(f"Leiden clustering completed with resolution {resolution_parameter:.4f}, "
                   f"resulting in {actual_n_clusters} clusters")

    # Build mapping from cohort_id to list of patient indices
    cohort_to_patients = {}
    for patient_idx, cohort_id in enumerate(cohort_assignments):
        if cohort_id not in cohort_to_patients:
            cohort_to_patients[cohort_id] = []
        cohort_to_patients[cohort_id].append(patient_idx)

    if logger:
        logger.success(f"Successfully created {actual_n_clusters} patient cohorts using Leiden algorithm")

    # Compute cohort centroids (average embeddings for each cohort)

    # Get all train embeddings: shape (dataset_size, embedding_dim)
    train_embeddings = train_dataset[emd_key]

    # Get sorted cohort IDs to ensure consistent ordering
    sorted_cohort_ids = sorted(cohort_to_patients.keys())
    n_clusters = len(sorted_cohort_ids)

    # Check if cohort IDs are continuous (0-indexed)
    if sorted_cohort_ids != list(range(n_clusters)):
        logger.warning(f"Cohort IDs are not continuous: {sorted_cohort_ids}. "
                      f"Will create array with max(cohort_id)+1 rows.")
        max_cohort_id = max(sorted_cohort_ids)
        n_clusters = max_cohort_id + 1

    # Get embedding dimension
    if isinstance(train_embeddings, torch.Tensor):
        embedding_dim = train_embeddings.shape[1]
    else:
        embedding_dim = train_embeddings.shape[1]

    # Initialize centroids array
    centroids = np.zeros((n_clusters, embedding_dim), dtype=np.float32)

    # Compute centroid for each cohort
    for cohort_id, patient_indices in cohort_to_patients.items():
        # Get embeddings for all patients in this cohort
        cohort_embeddings = train_embeddings[patient_indices]  # Shape: (cohort_size, embedding_dim)

        # Convert to numpy if needed
        if isinstance(cohort_embeddings, torch.Tensor):
            cohort_embeddings = cohort_embeddings.cpu().numpy()

        # Compute average embedding (centroid)
        centroid = np.mean(cohort_embeddings, axis=0)  # Shape: (embedding_dim,)
        centroids[cohort_id] = centroid

    if logger:
        logger.success(f"Successfully computed centroids for {len(cohort_to_patients)} cohorts")

    return cohort_assignments, cohort_to_patients, centroids, g

def visualize_graph_with_clusters(
    g: ig.Graph,
    cohort_assignments: np.ndarray,
    save_path: Optional[str] = None,
    logger=None,
    max_nodes: int = 900,
    layout: str = 'fr'
) -> None:
    '''
    Set cohort visualization attributes and return the graph.
    Args:
        g: igraph Graph object
        cohort_assignments: numpy array of shape (dataset_size,) where each element is the cohort ID
        save_path: compatibility argument; this helper does not write image files.
        logger: optional logger instance
        max_nodes: maximum number of nodes to visualize (for large graphs, will sample)
        layout: layout algorithm to use ('fr', 'kk', 'lgl', etc.)
    '''
    if logger is None:
        logger = get_logger("GraphWalker")

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    num_nodes = g.vcount()
    num_edges = g.ecount()
    num_clusters = len(np.unique(cohort_assignments))

    logger.info(f"Visualizing graph with {num_nodes} nodes, {num_edges} edges, and {num_clusters} clusters")

    # For large graphs, sample nodes for visualization
    original_num_nodes = num_nodes
    if num_nodes > max_nodes:
        logger.warning(f"Graph has {num_nodes} nodes, sampling {max_nodes} nodes for visualization")
        # Sample nodes while preserving cluster distribution
        sampled_indices = []
        for cluster_id in np.unique(cohort_assignments):
            cluster_nodes = np.where(cohort_assignments == cluster_id)[0]
            sample_size = min(len(cluster_nodes), max_nodes // num_clusters)
            sampled = np.random.choice(cluster_nodes, size=sample_size, replace=False)
            sampled_indices.extend(sampled)

        sampled_indices = sorted(sampled_indices)  # Sort to maintain order

        # Create subgraph
        g = g.subgraph(sampled_indices)
        # Map original indices to new indices in subgraph
        cohort_assignments = cohort_assignments[sampled_indices]
        num_nodes = g.vcount()
        logger.info(f"Sampled {num_nodes} nodes for visualization (from {original_num_nodes} nodes)")

    # Set vertex colors based on cluster assignments
    # Generate distinct colors for each cluster
    unique_clusters = np.unique(cohort_assignments)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Map cluster IDs to colors
    cluster_to_color = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
    vertex_colors = [mcolors.rgb2hex(cluster_to_color[cohort_assignments[i]]) for i in range(num_nodes)]

    # Set vertex attributes
    g.vs['color'] = vertex_colors
    g.vs['cluster'] = cohort_assignments.tolist()

    # Set edge attributes (make edges semi-transparent)
    if 'weight' in g.edge_attributes():
        # Normalize edge weights for visualization
        weights = np.array(g.es['weight'])
        if len(weights) > 0:
            min_weight, max_weight = weights.min(), weights.max()
            if max_weight > min_weight:
                normalized_weights = (weights - min_weight) / (max_weight - min_weight)
                # Map to edge width (1-3)
                g.es['width'] = 1 + 2 * normalized_weights
            else:
                g.es['width'] = 1.0
        else:
            g.es['width'] = 1.0
    else:
        g.es['width'] = 1.0

    # Set vertex size (larger for better cluster visibility)
    if num_nodes > 500:
        g.vs['size'] = 12
    elif num_nodes > 200:
        g.vs['size'] = 18
    else:
        g.vs['size'] = 25

    # Compute layout
    logger.info(f"Computing {layout} layout...")
    try:
        if layout == 'fr':
            layout_result = g.layout('fr')
        elif layout == 'kk':
            layout_result = g.layout('kk')
        elif layout == 'lgl':
            layout_result = g.layout('lgl')
        else:
            layout_result = g.layout('fr')
    except Exception as e:
        logger.warning(f"Failed to compute {layout} layout, using default: {e}")
        layout_result = g.layout('auto')

    return g

def select_graph_walker_examples_for_single_patient(
    args, patient_example, train_dataset, graph,
    centroids, cohort_to_patients, num_examples,
    emb_key: str = 'smart_embedding', mode:str='frontiers-full-greedy', logger=None,
    vllm_model: LLM = None
) -> List[Dict[str, Any]]:

    top_l_cohorts = getattr(args, 'graph_walker_top_l_cohorts', 3)  # top-L candidate cohorts

    # find top-L candidate cohorts for the test patient
    test_patient_embedding = patient_example[emb_key]

    candidate_cohort_ids = _find_top_l_candidate_cohorts(
        test_patient_embedding, centroids, top_l_cohorts
    )

    if mode == 'random':
        candidate_patient_indices = []
        for cohort_id in candidate_cohort_ids:
            cohort_patient_indices = cohort_to_patients.get(cohort_id, [])
            candidate_patient_indices.extend(cohort_patient_indices)
        selected_patient_indices = np.random.choice(candidate_patient_indices, size=min(num_examples, len(candidate_patient_indices)), replace=False)
    elif mode == 'frontiers-full-greedy':
        top_k_per_cohort = getattr(args, 'graph_walker_top_k_per_cohort', 3)  # top-k per cohort (default: 3)
        # build frontiers set by selecting top-k patients from each candidate cohort
        frontiers = _build_frontiers_from_cohorts(
            test_patient_embedding, train_dataset, candidate_cohort_ids,
            cohort_to_patients, top_k_per_cohort, emb_key=emb_key
        )
        # Greedy search over the current frontier.
        selected_patient_indices = _greedy_graph_walk(
            args, patient_example, train_dataset, graph,
            num_examples, vllm_model, parallel_batch_size_for_cal_greedy_score = args.graph_walker_parallel_batch_size_for_cal_greedy_score,
            frontiers=frontiers, logger=logger
        )
    else:
        raise ValueError(f"Invalid mode for graph walker: {mode}")

    ICL_EXAMPLES_LIST = []
    for patient_idx in selected_patient_indices:
        example_dict = {
            'node_index': int(patient_idx),
            'detail': train_dataset['detail'][patient_idx],
            'label': train_dataset['y'][patient_idx],
        }
        # Add smart_logits if available
        if 'smart_logits' in train_dataset:
            example_dict['smart_logits'] = train_dataset['smart_logits'][patient_idx]
        ICL_EXAMPLES_LIST.append(example_dict)
    return ICL_EXAMPLES_LIST


def _find_top_l_candidate_cohorts(
    test_patient_embedding, centroids, top_l_cohorts
) -> List[int]:
    '''
    Find top-L candidate cohorts for a test patient based on similarity to cohort centroids
    Args:
        test_patient_embedding: embedding of the test patient (torch.Tensor or np.ndarray)
        centroids: numpy array of shape (n_clusters, embedding_dim) containing cluster centers
        top_l_cohorts: number of top candidate cohorts to return
    Returns:
        List of cohort IDs (indices) sorted by similarity (most similar first)
    '''
    # Convert test patient embedding to numpy if needed
    if isinstance(test_patient_embedding, torch.Tensor):
        test_embedding_np = test_patient_embedding.cpu().numpy()
        if test_embedding_np.ndim == 1:
            test_embedding_np = test_embedding_np.reshape(1, -1)
    else:
        test_embedding_np = np.array(test_patient_embedding)
        if test_embedding_np.ndim == 1:
            test_embedding_np = test_embedding_np.reshape(1, -1)

    # Normalize embeddings for cosine similarity
    test_norm = test_embedding_np / (np.linalg.norm(test_embedding_np, axis=1, keepdims=True) + 1e-8)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity: (1, embedding_dim) @ (embedding_dim, n_clusters) -> (1, n_clusters)
    similarities = np.dot(test_norm, centroids_norm.T).squeeze(0)  # Shape: (n_clusters,)

    # Get top-L most similar cohort indices
    top_l_cohorts = min(top_l_cohorts, len(similarities))
    top_l_indices = np.argsort(similarities)[::-1][:top_l_cohorts].tolist()

    return top_l_indices

def _build_frontiers_from_cohorts(
    test_patient_embedding, train_dataset, candidate_cohort_ids: List[int],
    cohort_to_patients: Dict[int, List[int]], top_k_per_cohort: int = 3, emb_key: str = 'smart_embedding'
) -> Set[int]:
    '''
    Build frontiers set by selecting top-k most similar patients from each candidate cohort
    Args:
        test_patient_embedding: embedding of the test patient (torch.Tensor or np.ndarray)
        train_dataset: the train dataset
        candidate_cohort_ids: list of candidate cohort IDs
        cohort_to_patients: dict mapping cohort_id -> list of patient indices in that cohort
        top_k_per_cohort: number of top-k patients to select from each cohort (default: 3)
    Returns:
        Set of patient indices (frontiers) from all candidate cohorts
    '''
    frontiers = set()

    # Get all train embeddings
    train_embeddings = train_dataset[emb_key]

    # Ensure test_patient_embedding is 2D for similarity computation
    if isinstance(test_patient_embedding, torch.Tensor):
        test_emb = test_patient_embedding
        if test_emb.dim() == 1:
            test_emb = test_emb.unsqueeze(0)
    else:
        test_emb = torch.tensor(test_patient_embedding)
        if test_emb.dim() == 1:
            test_emb = test_emb.unsqueeze(0)

    # Normalize test embedding
    test_norm = torch.nn.functional.normalize(test_emb, p=2, dim=1)

    # For each candidate cohort, find top-k most similar patients
    for cohort_id in candidate_cohort_ids:
        # Get all patient indices in this cohort
        cohort_patient_indices = cohort_to_patients.get(cohort_id, [])

        if not cohort_patient_indices:
            continue

        # Get embeddings for patients in this cohort
        cohort_embeddings = train_embeddings[cohort_patient_indices]  # Shape: (cohort_size, embedding_dim)

        # Normalize cohort embeddings
        cohort_norm = torch.nn.functional.normalize(cohort_embeddings, p=2, dim=1)

        # Compute cosine similarity: (1, embedding_dim) @ (embedding_dim, cohort_size) -> (1, cohort_size)
        similarities = torch.mm(test_norm, cohort_norm.t()).squeeze(0)  # Shape: (cohort_size,)

        # Get top-k most similar indices within this cohort
        k = min(top_k_per_cohort, len(cohort_patient_indices))
        topk_values, topk_local_indices = torch.topk(similarities, k=k, dim=0)

        # Convert to list and get actual patient indices
        if isinstance(topk_local_indices, torch.Tensor):
            topk_local_indices = topk_local_indices.cpu().tolist()
        else:
            topk_local_indices = [topk_local_indices] if not isinstance(topk_local_indices, list) else topk_local_indices

        # Map local indices to actual patient indices
        for local_idx in topk_local_indices:
            actual_patient_idx = cohort_patient_indices[local_idx]
            frontiers.add(actual_patient_idx)

    return frontiers



def _greedy_graph_walk(
    args, test_patient_example, train_dataset, graph,
    max_examples: int, vllm_model: LLM,
    parallel_batch_size_for_cal_greedy_score: int,
    frontiers: Optional[Set[int]] = None, logger=None,
) -> List[int]:
    """Recompute every frontier candidate's marginal gain after each selection."""
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    frontier = set(frontiers or ())
    if max_examples == 0 or not frontier:
        return []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = []
    early_stop = not getattr(args, 'graph_walker_no_early_stop', False)

    def evaluate(compositions):
        values = _compute_cross_entropy_for_examples_batch(
            args, test_patient_example, train_dataset, compositions,
            vllm_model, device, parallel_batch_size_for_cal_greedy_score,
        )
        if len(values) != len(compositions) or not np.isfinite(values).all():
            raise ValueError("Entropy scoring returned missing or non-finite values")
        return values

    current_ce = evaluate([[]])[0]
    while frontier and len(selected) < max_examples:
        candidates = sorted(frontier)
        ces = evaluate([selected + [node] for node in candidates])
        best_ce, best = min(zip(ces, candidates))
        if early_stop and current_ce - best_ce <= 0:
            break
        selected.append(best)
        current_ce = best_ce
        frontier.remove(best)
        frontier.update(node for node, _ in graph.get(best, ()) if node not in selected)
    return selected


def _compute_cross_entropy_for_examples_batch(
    args, test_patient_example, train_dataset, example_node_indices_list: List[List[int]],
    vllm_model: LLM, device: torch.device, parallel_batch_size_for_cal_greedy_score: int
) -> List[float]:
    '''
    Batch compute cross-entropy loss for test patient with multiple ICL example lists
    Uses vLLM for acceleration
    Args:
        args: the arguments
        test_patient_example: the test patient example
        train_dataset: the train dataset
        example_node_indices_list: list of example node indices lists, each for one candidate
        vllm_model: vLLM model instance for computing cross-entropy
        device: device to run the model on
        parallel_batch_size_for_cal_greedy_score: batch size for parallel processing when computing cross-entropy
    Returns:
        List of cross-entropy losses (one for each example_node_indices)
    '''
    tokenizer = vllm_model.get_tokenizer()
    from llms.prompt_format import format_model_prompt
    prompts, mask_lengths, test_lengths = [], [], []
    for indices in example_node_indices_list:
        prompt, start, end = _build_prompt_from_examples(
            args, test_patient_example, train_dataset, indices, return_target_span=True
        )
        prompt, start, end = format_model_prompt(
            args, tokenizer, prompt, (start, end),
            enable_thinking=getattr(args, 'vllm_enable_thinking', False))
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        positions = [i for i, (lo, hi) in enumerate(encoded['offset_mapping'])
                     if hi > start and lo < end and hi > lo]
        if not positions:
            raise ValueError("The target query has no scorable tokens")
        prompts.append(prompt)
        mask_lengths.append(positions[0])
        test_lengths.append(positions[-1] + 1)
    return _compute_cross_entropy_loss_vllm(
        prompts, vllm_model, tokenizer,
        batch_size=parallel_batch_size_for_cal_greedy_score,
        mask_lengths=mask_lengths, test_lengths=test_lengths,
    )


def _compute_cross_entropy_for_examples(
    args, test_patient_example, train_dataset, example_node_indices: List[int],
    vllm_model: LLM, device: torch.device, parallel_batch_size_for_cal_greedy_score: int
) -> float:
    '''
    Compute cross-entropy loss for test patient with given ICL examples
    Uses vLLM for acceleration
    Args:
        args: the arguments
        test_patient_example: the test patient example
        train_dataset: the train dataset
        example_node_indices: list of node indices to use as ICL examples
        vllm_model: vLLM model instance for computing cross-entropy
        device: device to run the model on
        parallel_batch_size_for_cal_greedy_score: batch size for parallel processing when computing cross-entropy
    Returns:
        Cross-entropy loss (float)
    '''
    # Use batch version with a single example list
    ce_losses = _compute_cross_entropy_for_examples_batch(
        args, test_patient_example, train_dataset, [example_node_indices],
        vllm_model, device, parallel_batch_size_for_cal_greedy_score
    )
    return ce_losses[0]


def _build_prompt_from_examples(
    args, test_patient_example, train_dataset, example_node_indices: List[int],
    return_target_start: bool = False, return_target_span: bool = False,
):
    """Build the same patient query used for final inference."""
    from prompt.EHR_prompt.common import build_ehr_prompt
    examples = []
    for node in example_node_indices:
        example = {key: values[node] for key, values in train_dataset.items()}
        example['label'] = train_dataset['y'][node]
        examples.append(example)
    result = build_ehr_prompt(
        test_patient_example, args.dataset, examples,
        add_smart_logits=getattr(args, 'graph_walker_add_smart_logits', False),
        add_smart_logits_for_test_example=getattr(args, 'graph_walker_add_smart_logits_for_test_example', False),
        return_target_span=True)
    if return_target_span:
        return result
    return result[:2] if return_target_start else result[0]


def _compute_cross_entropy_loss_vllm(
    prompts: List[str], vllm_model: LLM, tokenizer: AutoTokenizer,
    batch_size: int = 4, mask_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None,
) -> List[float]:
    """Mean target-token NLL, using identical token IDs for scoring and masks.

    Scores input tokens rather than the output-label distribution.
    Target labels are excluded from the prompt.
    """
    from vllm import SamplingParams
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(x is not None and len(x) != len(prompts) for x in (mask_lengths, test_lengths)):
        raise ValueError("Token spans must have one entry per prompt")
    all_losses = []
    for start in range(0, len(prompts), batch_size):
        token_batches = [tokenizer(p, add_special_tokens=False)['input_ids']
                         for p in prompts[start:start + batch_size]]
        outputs = vllm_model.generate(
            [{'prompt_token_ids': ids} for ids in token_batches],
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1),
            use_tqdm=False,
        )
        if len(outputs) != len(token_batches):
            raise ValueError("vLLM did not return one score per prompt")
        for j, (output, ids) in enumerate(zip(outputs, token_batches)):
            index = start + j
            lo = mask_lengths[index] if mask_lengths is not None else 1
            hi = test_lengths[index] if test_lengths is not None else len(ids)
            if not 0 <= lo < hi <= len(ids):
                raise ValueError("Invalid target-token span")
            if list(output.prompt_token_ids) != list(ids):
                raise ValueError("vLLM token IDs differ from the scoring mask")
            logprobs = output.prompt_logprobs
            if logprobs is None or len(logprobs) != len(ids):
                raise ValueError("vLLM returned missing or incomplete prompt logprobs")
            losses = []
            # Position i is the probability of input_ids[i]; do not shift again.
            for pos in range(max(1, lo), hi):
                entry = logprobs[pos]
                if entry is None or ids[pos] not in entry:
                    raise ValueError(f"Missing observed-token logprob at position {pos}")
                value = -float(entry[ids[pos]].logprob)
                if not np.isfinite(value):
                    raise ValueError("Non-finite observed-token logprob")
                losses.append(value)
            if not losses:
                raise ValueError("The target query has no scorable tokens")
            all_losses.append(sum(losses) / len(losses))
    return all_losses


def _compute_cross_entropy_loss(
    prompts: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 4,
    mask_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None
) -> List[float]:
    '''
    Compute cross-entropy loss for each prompt using a language model
    Args:
        prompts: List of prompt strings
        model_name: Path to the language model
        device: Device to run the model on
        batch_size: Batch size for parallel processing prompts
        mask_lengths: List of ICE token lengths for each prompt (optional)
        test_lengths: List of ICE+test token lengths for each prompt (optional)
    Returns:
        List of cross-entropy losses for each prompt
    '''
    if model_name is None:
        raise ValueError("llm_local_path must be specified in args for GraphWalker method")

    # Load tokenizer and model with caching
    tokenizer = _get_tokenizer(model_name, device)

    # Load model with caching and memory optimizations
    if model_name not in _metric_model_cache:
        try:
            # Use FP16/BF16 to reduce memory usage
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            # Try bfloat16 if available (better numerical stability)
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16

            # Load model with memory optimizations
            use_device_map = torch.cuda.is_available()
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if use_device_map else None,
            )
            # Only move to device if device_map wasn't used
            if not use_device_map:
                model.to(device)
            model.eval()
            _metric_model_cache[model_name] = model
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_name}: {e}")

    model = _metric_model_cache[model_name]

    all_losses = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_mask_lengths = mask_lengths[i:i + batch_size] if mask_lengths else None
        batch_test_lengths = test_lengths[i:i + batch_size] if test_lengths else None

        # Tokenize
        inputs = tokenizer(batch_prompts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Compute cross-entropy loss
        # Shift logits and labels for next-token prediction
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)
        loss = loss_fct(shift_logits_flat, shift_labels_flat).view(shift_labels.size())

        # Apply mask if provided (only compute loss for test part)
        if batch_mask_lengths is not None and batch_test_lengths is not None:
            mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
            for j in range(len(mask)):
                mask_start = batch_mask_lengths[j]
                mask_end = batch_test_lengths[j]
                # Ensure indices are within bounds
                mask_start = min(mask_start, mask.size(1))
                mask_end = min(mask_end, mask.size(1))
                if mask_start < mask_end:
                    mask[j, mask_start:mask_end] = 1
            loss = loss * mask

        # Sum over sequence length for each prompt and move to CPU immediately
        ce_loss = torch.sum(loss, dim=1).cpu().tolist()
        all_losses.extend(ce_loss)

        # Clear intermediate tensors to free GPU memory
        del outputs, shift_logits, shift_labels, shift_logits_flat, shift_labels_flat, loss
        if batch_mask_lengths is not None and batch_test_lengths is not None:
            del mask
        del inputs

        # Periodically clear GPU cache to avoid fragmentation
        if (i + batch_size) % (batch_size * 4) == 0 or i + batch_size >= len(prompts):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_losses


def _get_tokenizer(model_name: str, device: torch.device) -> AutoTokenizer:
    '''Get or create tokenizer with caching'''
    if model_name not in _metric_tokenizer_cache:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "right"

            _metric_tokenizer_cache[model_name] = tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {model_name}: {e}")

    return _metric_tokenizer_cache[model_name]


def clear_graph_walker_cache():
    '''
    Clear cached models and tokenizers to release GPU memory
    '''
    global _metric_model_cache, _metric_tokenizer_cache

    # Clear models from GPU memory
    for model_name, model in _metric_model_cache.items():
        if model is not None:
            # Move model to CPU and delete
            try:
                model.cpu()
                del model
            except Exception as e:
                pass  # Silently handle cleanup errors

    # Clear caches
    _metric_model_cache.clear()
    _metric_tokenizer_cache.clear()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
