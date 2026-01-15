'''
    Graph walker method implementation Naive Edition v4

    Algorithm Summary:
    - 1. build a KNN graph from the train dataset via smart embedding 
    similarity
    - 2. using Leiden to fined patient cohorts
    - 3. test patient cohorts:  using majority voting to find top-L candicate cohorts
    - 4. In each cohort, select the top-K examples similar/different to the test patient as anchor points
    - 5. frontiers lazy greedy search

    # TODO: vllm support

'''
from typing import List, Dict, Any, Optional, Set, Tuple
import torch
import numpy as np
import heapq
import time
import os
import gc
from pathlib import Path
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logger import get_logger
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from vllm import LLM, SamplingParams

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
    
    # Visualize the graph (disabled for open source release)
    # visualize_graph_with_clusters(
    #         igraph_g, cohort_assignments, 
    #         save_path=getattr(args, 'graph_visualization_path', './graph_visualization.png'),
    #         layout='fr',
    #         logger=logger
    #     )

    # load vllm model
    vllm_model = _load_vllm_model(args, logger=logger)

    # select ICL examples for all test pati
    patient_num = len(test_dataset['detail'])
    progress = logger.create_progress("Selecting GraphWalker examples for all test patients", patient_num)
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
    
    # Release vLLM model memory
    _release_vllm_model(vllm_model, logger=logger)
    
    return ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS

def _load_vllm_model(args, logger=None)->LLM:
    
    # detect gpu count
    gpu_count = torch.cuda.device_count()
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
                   "swap_space": 4}  # Use disk swap space in GB if shared memory is insufficient
    # initialize vllm model with retry logic
    logger.info(f"Initializing vLLM with kwargs: {vllm_kwargs}")
    vllm_model = None
    # Normal initialization
    vllm_model = LLM(**vllm_kwargs)
    logger.success("vLLM model initialized successfully")
    return vllm_model

def _release_vllm_model(vllm_model: LLM, logger=None):
    '''
    Release vLLM model memory
    Args:
        vllm_model: vLLM model instance to release
        logger: logger instance (optional)
    '''
    if vllm_model is None:
        return
    
    try:
        if logger:
            logger.info("Releasing vLLM model memory...")
        
        # Delete the model object
        # vLLM will handle internal resource cleanup when the object is deleted
        del vllm_model
        
        # Force garbage collection to ensure Python objects are freed
        gc.collect()
        
        # Clear GPU cache to free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if logger:
            logger.success("vLLM model memory released successfully")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to release vLLM model memory: {e}")

def _build_graph(args, train_dataset, neighbor_num: int = 3, logger=None, emd_key: str = 'smart_embedding'):
    '''
    Build the graph from the train dataset using EHR model embedding
    Each node is a patient example, and neighbors are top-k embedding-similar patients
    Args:
        args: the arguments
        train_dataset: the train dataset
    Returns:
        the graph: dict where graph[i] = [(neighbor_idx, weight), ...]
                  weight is the cosine similarity between node i and neighbor
    '''
    logger.info("building graph...")
    graph = {}
    # Get all train embeddings: shape (dataset_size, embedding_dim)
    train_embeddings = train_dataset[emd_key]
    dataset_size = train_embeddings.shape[0]
    
    k = neighbor_num
    # Ensure k is at most dataset_size - 1 (excluding self)
    k = min(k, dataset_size - 1)
    
    # Normalize embeddings for cosine similarity
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)  # (dataset_size, embedding_dim)
    
    # Compute pairwise cosine similarity matrix
    # (dataset_size, embedding_dim) @ (embedding_dim, dataset_size) -> (dataset_size, dataset_size)
    similarity_matrix = torch.mm(train_norm, train_norm.t())  # (dataset_size, dataset_size)
    
    # For each patient, find top-k neighbors (excluding self)
    for i in range(dataset_size):
        # Get similarities for patient i with all other patients
        similarities = similarity_matrix[i].clone()  # (dataset_size,) - clone to avoid modifying original
        
        # Set self-similarity to -inf to exclude self from top-k
        similarities[i] = float('-inf')
        
        # Get top-k most similar indices and their values
        topk_values, topk_indices = torch.topk(similarities, k=k, dim=0)
        
        # Convert to list of tuples (neighbor_idx, weight)
        if isinstance(topk_indices, torch.Tensor):
            neighbor_indices = topk_indices.cpu().tolist()
            neighbor_weights = topk_values.cpu().tolist()
        else:
            neighbor_indices = [topk_indices]
            neighbor_weights = [topk_values]
        
        # Ensure neighbors is a list
        if not isinstance(neighbor_indices, list):
            neighbor_indices = [neighbor_indices]
            neighbor_weights = [neighbor_weights]
        
        # Store as list of (neighbor_idx, weight) tuples
        graph[i] = list(zip(neighbor_indices, neighbor_weights))
    
    logger.success("graph built successfully")
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
            resolution_parameter = 1.0  # Default resolution
    
    if random_state is None:
        if args is not None and hasattr(args, 'seed'):
            random_state = args.seed
        else:
            random_state = 42  # Default seed
    
    logger.info(f"Performing Leiden clustering with resolution={resolution_parameter}, seed={random_state}...")
    
    # Get dataset size from graph (graph contains all nodes from 0 to dataset_size-1)
    dataset_size = len(graph)
    
    # Build igraph Graph object from the adjacency list with edge weights
    # Create an undirected graph
    g = ig.Graph(directed=False)
    g.add_vertices(dataset_size)
    
    # Collect edges with weights
    # Use a dict to store edge weights, handling duplicates by taking the average
    edge_weights_dict = {}
    edges_list = []
    
    for node, neighbors in graph.items():
        for neighbor_idx, weight in neighbors:
            # Create edge tuple with smaller index first to handle undirected graph
            edge = (min(node, neighbor_idx), max(node, neighbor_idx))
            
            # If edge already exists, average the weights (since it's undirected, same edge may appear twice)
            if edge in edge_weights_dict:
                # Average the weights if edge appears multiple times
                edge_weights_dict[edge] = (edge_weights_dict[edge] + weight) / 2.0
            else:
                edge_weights_dict[edge] = weight
                edges_list.append(edge)
    
    # Add edges to graph
    if edges_list:
        g.add_edges(edges_list)
        
        # Set edge weights
        weights = [edge_weights_dict[edge] for edge in edges_list]
        g.es['weight'] = weights
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Perform Leiden clustering with weighted edges
    # RBERVertexPartition supports weighted graphs and resolution_parameter
    weights_list = g.es['weight'] if 'weight' in g.edge_attributes() else None
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBERVertexPartition,
        resolution_parameter=resolution_parameter,
        seed=random_state,
        weights=weights_list
    )
    
    # Get cluster assignments
    cohort_assignments = np.array(partition.membership)  # Shape: (dataset_size,)
    actual_n_clusters = len(partition)
    
    if logger:
        logger.info(f"Leiden clustering completed with resolution {resolution_parameter:.4f}, "
                   f"resulting in {actual_n_clusters} clusters")
        if 'weight' in g.edge_attributes():
            logger.info(f"Using weighted edges for Leiden clustering")
    
    # Build mapping from cohort_id to list of patient indices
    cohort_to_patients = {}
    for patient_idx, cohort_id in enumerate(cohort_assignments):
        if cohort_id not in cohort_to_patients:
            cohort_to_patients[cohort_id] = []
        cohort_to_patients[cohort_id].append(patient_idx)
    
    # Log statistics (simplified for open source)
    # cohort_sizes = {cohort_id: len(patients) for cohort_id, patients in cohort_to_patients.items()}
    # logger.info(f"Leiden clustering completed. Cohort sizes: {cohort_sizes}")
    if logger:
        logger.success(f"Successfully created {actual_n_clusters} patient cohorts using Leiden algorithm")
    
    # Compute cohort centroids (average embeddings for each cohort)
    # logger.info("Computing cohort centroids (average embeddings)...")
    
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
    # logger.info(f"Cohort centroids computed: {len(cohort_to_patients)} centroids, each with dimension {embedding_dim}")
    # logger.info(f"Centroids array shape: {centroids.shape}")
    
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
    Visualize the graph with different colors for different clusters/cohorts
    Args:
        g: igraph Graph object
        cohort_assignments: numpy array of shape (dataset_size,) where each element is the cohort ID
        save_path: optional path to save the visualization. If None, will try to use igraph's plot
        logger: optional logger instance
        max_nodes: maximum number of nodes to visualize (for large graphs, will sample)
        layout: layout algorithm to use ('fr', 'kk', 'lgl', etc.)
    '''
    if logger is None:
        logger = get_logger("GraphWalker")
    
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
    
    # Graph visualization saving disabled for open source release
    # if save_path:
    #     try:
    #         # Convert to absolute path and ensure directory exists
    #         save_path = os.path.abspath(save_path)
    #         save_dir = os.path.dirname(save_path)
    #         if save_dir and not os.path.exists(save_dir):
    #             os.makedirs(save_dir, exist_ok=True)
    #             logger.info(f"Created directory: {save_dir}")
    #         
    #         logger.info(f"Saving graph visualization to: {save_path}")
    #         ig.plot(g, save_path, layout=layout_result, 
    #                vertex_color=vertex_colors,
    #                vertex_size=g.vs['size'],
    #                edge_width=g.es['width'],
    #                bbox=(1600, 1200))
    #         logger.success(f"Graph visualization saved successfully!")
    #         logger.success(f"Full path: {save_path}")
    #         logger.info(f"Current working directory: {os.getcwd()}")
    #     except Exception as e:
    #         logger.warning(f"Failed to save using igraph plot (may need cairo): {e}")
    #         logger.info("You can still visualize the graph using igraph's interactive plot")
    
    return g

def select_graph_walker_examples_for_single_patient(
    args, patient_example, train_dataset, graph,
    centroids, cohort_to_patients, num_examples, 
    emb_key: str = 'smart_embedding', mode:str='random', logger=None,
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
        selected_patient_indices = np.random.choice(candidate_patient_indices, size=num_examples, replace=False)
    elif mode == 'frontiers-lazy-greedy':
        top_k_per_cohort = getattr(args, 'graph_walker_top_k_per_cohort', 2)  # top-k per cohort (default: 2)
        # build frontiers set by selecting top-k patients from each candidate cohort
        frontiers = _build_frontiers_from_cohorts(
            test_patient_embedding, train_dataset, candidate_cohort_ids,
            cohort_to_patients, top_k_per_cohort
        )        
        # Debug logging disabled for open source
        # if logger:
        #     logger.debug(f"Built frontiers set with {len(frontiers)} patients from {len(candidate_cohort_ids)} cohorts")
            
        # lazy greedy search on frontiers
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
    cohort_to_patients: Dict[int, List[int]], top_k_per_cohort: int = 2, emb_key: str = 'smart_embedding'
) -> Set[int]:
    '''
    Build frontiers set by selecting top-k most similar patients from each candidate cohort
    Args:
        test_patient_embedding: embedding of the test patient (torch.Tensor or np.ndarray)
        train_dataset: the train dataset
        candidate_cohort_ids: list of candidate cohort IDs
        cohort_to_patients: dict mapping cohort_id -> list of patient indices in that cohort
        top_k_per_cohort: number of top-k patients to select from each cohort (default: 2)
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
    parallel_batch_size_for_cal_greedy_score: int, frontiers: Optional[Set[int]] = None, logger=None
) -> List[int]:
    '''
    Lazy greedy graph walk algorithm based on cone score
    Uses a priority queue to avoid evaluating all candidates in each iteration.
    Performs greedy search on frontiers set, and expands frontiers by adding neighbors of selected nodes.
    Args:
        args: the arguments
        test_patient_example: the test patient example
        train_dataset: the train dataset
        graph: the graph built from train dataset
        max_examples: maximum number of examples to select
        vllm_model: vllm model for computing cross-entropy
        parallel_batch_size_for_cal_greedy_score: batch size for parallel
            computing cross-entropy for greedy score
        frontiers: initial set of frontier node indices
            (from candidate cohorts)
        logger: optional logger instance
    Returns:
        List of selected node indices
    '''
    selected_nodes = []  # List of selected node indices
    
    # Initialize frontiers set (make a copy to avoid modifying the original)
    if frontiers is None:
        frontiers = set()
    
    # Ensure frontiers is a set
    frontiers = set(frontiers)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compute baseline cross-entropy (with no examples)
    baseline_ce = _compute_cross_entropy_for_examples(
        args, test_patient_example, train_dataset, [], 
        vllm_model, device, parallel_batch_size_for_cal_greedy_score
    )
    
    current_ce = baseline_ce
    
        # Debug logging disabled for open source
        # if logger:
        #     logger.debug(f"Starting lazy greedy search with {len(frontiers)} initial frontier nodes")
    
    # Initialize priority queue: (-score, node_id, cached_score, cached_ce)
    # Use negative score because heapq is a min-heap, but we want max score
    # cached_score and cached_ce are the last computed values for this node
    priority_queue = []
    node_cache = {}  # node_id -> (cached_score, cached_ce)
    
    # Initialize queue with all frontier nodes
    # For lazy greedy, we first compute initial scores for all candidates
    initial_candidates = list(frontiers)
    if initial_candidates:
        candidate_example_lists = [
            selected_nodes + [candidate] for candidate in initial_candidates
        ]
        candidate_ces = _compute_cross_entropy_for_examples_batch(
            args, test_patient_example, train_dataset, candidate_example_lists,
            vllm_model, device, parallel_batch_size_for_cal_greedy_score
        )
        
        for candidate, new_ce in zip(initial_candidates, candidate_ces):
            score = current_ce - new_ce
            node_cache[candidate] = (score, new_ce)
            heapq.heappush(priority_queue, (-score, candidate))
    
    # Lazy greedy search: iterate until we have enough examples or no more candidates
    step_num = 0
    max_steps = max_examples * 5  # Safety limit: allow at most 5x max_examples steps
    while len(selected_nodes) < max_examples and step_num < max_steps:
        step_num += 1
        step_start_time = time.time()
        previous_ce = current_ce  # Store previous CE for delta calculation
        
        if not priority_queue:
            # No more candidates, stop
            # if logger:
            #     logger.debug(f"No more candidates in queue. Selected {len(selected_nodes)} nodes.")
            break
        
        # Lazy greedy: repeatedly check top element until we find one that's still optimal
        best_candidate = None
        best_score = float('-inf')
        best_new_ce = current_ce
        
        # Clean up queue: remove nodes that are already selected
        # This prevents infinite loops when all remaining nodes are already selected
        cleaned_queue = []
        for neg_score, node in priority_queue:
            if node not in selected_nodes:
                cleaned_queue.append((neg_score, node))
        priority_queue = cleaned_queue
        heapq.heapify(priority_queue)
        
        # Add safety counter to prevent infinite loops
        max_iterations = max(len(priority_queue) * 2, 100) if priority_queue else 0  # Allow checking each node at most twice, minimum 100
        iteration_count = 0
        
        while priority_queue and iteration_count < max_iterations:
            iteration_count += 1
            # Get top element (highest cached score)
            neg_cached_score, candidate = heapq.heappop(priority_queue)
            cached_score = -neg_cached_score
            
            # Skip if already selected (shouldn't happen after cleanup, but double-check)
            if candidate in selected_nodes:
                continue
            
            # Recompute score for this candidate (since selected_nodes has changed)
            candidate_example_list = selected_nodes + [candidate]
            new_ce = _compute_cross_entropy_for_examples(
                args, test_patient_example, train_dataset, candidate_example_list,
                vllm_model, device, parallel_batch_size_for_cal_greedy_score
            )
            score = current_ce - new_ce
            
            # Update cache
            node_cache[candidate] = (score, new_ce)
            
            # Check if this candidate is still the best (lazy greedy condition)
            # If queue is empty or this score >= next best cached score, we can select it
            if not priority_queue:
                # This is the only candidate, select it
                best_candidate = candidate
                best_score = score
                best_new_ce = new_ce
                break
            else:
                # Check next best cached score
                next_neg_score, _ = priority_queue[0]
                next_cached_score = -next_neg_score
                
                if score >= next_cached_score:
                    # This candidate is still optimal, select it
                    best_candidate = candidate
                    best_score = score
                    best_new_ce = new_ce
                    break
                else:
                    # Re-insert with updated score
                    heapq.heappush(priority_queue, (-score, candidate))
        
        # Check if we exited due to max iterations (potential infinite loop)
        if iteration_count >= max_iterations and best_candidate is None:
            if logger:
                logger.warning(f"Reached max iterations ({max_iterations}) in lazy greedy search. "
                             f"Queue size: {len(priority_queue)}, Selected: {len(selected_nodes)}. "
                             f"This may indicate a potential infinite loop. Breaking.")
            break
        
        # Check stopping condition: if best candidate doesn't improve, stop
        if best_candidate is None or best_score <= 0:
            # if logger:
            #     logger.debug(f"No improving candidate found. Selected {len(selected_nodes)} nodes.")
            break
        
        # Add best candidate to selected nodes
        selected_nodes.append(best_candidate)
        current_ce = best_new_ce
        
        # Calculate delta (CE improvement)
        delta_ce = previous_ce - current_ce  # Positive delta means improvement (lower CE is better)
        
        # Remove selected node from cache and frontiers
        node_cache.pop(best_candidate, None)
        frontiers.discard(best_candidate)
        
        # Expand frontiers: add neighbors of the selected node
        neighbors = graph.get(best_candidate, [])
        new_neighbors = []
        for neighbor_tuple in neighbors:
            neighbor_idx = neighbor_tuple[0]  # Extract neighbor index from (neighbor_idx, weight) tuple
            if neighbor_idx not in selected_nodes and neighbor_idx not in frontiers:
                frontiers.add(neighbor_idx)
                new_neighbors.append(neighbor_idx)
        
        # Add new neighbors to priority queue
        if new_neighbors:
            candidate_example_lists = [
                selected_nodes + [candidate] for candidate in new_neighbors
            ]
            candidate_ces = _compute_cross_entropy_for_examples_batch(
                args, test_patient_example, train_dataset, candidate_example_lists,
                vllm_model, device, parallel_batch_size_for_cal_greedy_score
            )
            
            for candidate, new_ce in zip(new_neighbors, candidate_ces):
                score = current_ce - new_ce
                node_cache[candidate] = (score, new_ce)
                heapq.heappush(priority_queue, (-score, candidate))
        
        # Calculate time consumed for this step
        step_time = time.time() - step_start_time
        
        # Log statistics for this step (simplified for open source)
        # if logger:
        #     logger.info(
        #         f"Step {step_num}: Selected node {best_candidate} | "
        #         f"CE: {current_ce:.6f} | "
        #         f"Delta: {delta_ce:.6f} | "
        #         f"Time: {step_time:.3f}s | "
        #         f"Frontiers: {len(frontiers)} | "
        #         f"Selected: {len(selected_nodes)}/{max_examples}"
        #     )
    
    # Check if we exited due to max steps (potential infinite loop)
    if step_num >= max_steps and len(selected_nodes) < max_examples:
        if logger:
            logger.warning(f"Reached max steps ({max_steps}) in greedy graph walk. "
                         f"Selected {len(selected_nodes)}/{max_examples} nodes. "
                         f"This may indicate a potential infinite loop.")
    
    return selected_nodes


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
    # Build prompts for all candidates
    prompts = []
    for example_node_indices in example_node_indices_list:
        prompt = _build_prompt_from_examples(
            args, test_patient_example, train_dataset, example_node_indices
        )
        prompts.append(prompt)
    
    # Get tokenizer from vLLM model or from args
    model_path = args.llm_local_path
    tokenizer = _get_tokenizer(model_path, device)
    test_detail = test_patient_example['detail']
    
    # Calculate mask_length and test_length for each prompt
    mask_lengths = []
    test_lengths = []
    
    for prompt in prompts:
        # Find where test detail starts - everything before this is ICE part
        test_detail_start_marker = "Clinical Features Over Time:\n"
        test_detail_start_pos = prompt.find(test_detail_start_marker)
        if test_detail_start_pos != -1:
            # ICE part ends right before test detail starts
            ice_part = prompt[:test_detail_start_pos + len(test_detail_start_marker)]
        else:
            # Fallback: find test_detail content directly
            detail_start = prompt.find(test_detail)
            if detail_start != -1:
                ice_part = prompt[:detail_start]
            else:
                # Last resort: use everything before the final "Label:"
                ice_part = prompt[:prompt.rindex("Label:")]
        
        ice_tokenized = tokenizer(ice_part, return_tensors='pt', truncation=False, add_special_tokens=False)
        mask_length = len(ice_tokenized['input_ids'][0])
        
        # Find test part position (after "Label:" keyword at the end)
        key_word = "Label:"
        test_pos = prompt.rindex(key_word) + len(key_word)
        test_tokenized = tokenizer(prompt[:test_pos], return_tensors='pt', truncation=False, add_special_tokens=False)
        test_length = len(test_tokenized['input_ids'][0])
        
        mask_lengths.append(mask_length)
        test_lengths.append(test_length)
    
    # Use vLLM to compute cross-entropy loss (only on test part)
    ce_losses = _compute_cross_entropy_loss_vllm(
        prompts=prompts,
        vllm_model=vllm_model,
        tokenizer=tokenizer,
        batch_size=parallel_batch_size_for_cal_greedy_score,
        mask_lengths=mask_lengths,
        test_lengths=test_lengths
    )
    
    return ce_losses


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
    args, test_patient_example, train_dataset, example_node_indices: List[int]
) -> str:
    '''
    Build prompt from ICL examples and test patient
    Args:
        args: the arguments
        test_patient_example: the test patient example
        train_dataset: the train dataset
        example_node_indices: list of node indices to use as ICL examples
    Returns:
        Prompt string
    '''
    # Load prompt template and task description based on dataset
    from prompt.prompt_template import USERPROMPT_FEW_SHOT, TASK_DESCRIPTION, RESPONSE_FORMAT_ONLY_ANSWER
    dataset_name = args.dataset
    task_description = TASK_DESCRIPTION.get(dataset_name, '')
    response_format = RESPONSE_FORMAT_ONLY_ANSWER.get(dataset_name, '')
    
    # Get test patient information for prompt formatting
    X = test_patient_example['X']
    record_times = X[:, 0].astype(float)
    test_detail = test_patient_example['detail']
    
    # Build examples string
    examples_parts = []
    for idx, node_idx in enumerate(example_node_indices, 1):
        example_detail = train_dataset['detail'][node_idx]
        example_label = train_dataset['y'][node_idx]
        example_str = f"Example {idx}:\n{example_detail}\nLabel: {example_label}"
        examples_parts.append(example_str)
    
    example = "\n\n".join(examples_parts) if examples_parts else ""
    
    # Generate prompt using USERPROMPT_FEW_SHOT template
    prompt = USERPROMPT_FEW_SHOT.format(
        LENGTH=len(record_times),
        RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
        DETAIL=test_detail,
        RESPONSE_FORMAT=response_format,
        TASK_DESCRIPTION=task_description,
        EXAMPLE=example,
    )
    
    # Add "Label:" at the end for test prediction
    prompt += "\nLabel:"
    
    return prompt


def _compute_cross_entropy_loss_vllm(
    prompts: List[str],
    vllm_model: LLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    mask_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None
) -> List[float]:
    '''
    Compute cross-entropy loss for each prompt using vLLM (accelerated version)
    Args:
        prompts: List of prompt strings
        vllm_model: vLLM model instance
        tokenizer: Tokenizer instance
        batch_size: Batch size for parallel processing prompts
        mask_lengths: List of ICE token lengths for each prompt (optional)
        test_lengths: List of ICE+test token lengths for each prompt (optional)
    Returns:
        List of cross-entropy losses for each prompt
    '''
    from vllm import SamplingParams
    
    all_losses = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_mask_lengths = mask_lengths[i:i + batch_size] if mask_lengths else None
        batch_test_lengths = test_lengths[i:i + batch_size] if test_lengths else None
        
        # Tokenize prompts to get token IDs
        batch_token_ids = []
        for prompt in batch_prompts:
            tokenized = tokenizer(prompt, return_tensors='pt', truncation=False, add_special_tokens=False)
            token_ids = tokenized['input_ids'][0].tolist()
            batch_token_ids.append(token_ids)
        
        # Use vLLM to get prompt logprobs
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,  # 不采样新 token，只要 prompt_logprobs
            prompt_logprobs=1  # 获取所有 prompt token 的 logprobs
        )
        
        outputs = vllm_model.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        # Compute cross-entropy loss for each prompt
        for j, (output, token_ids) in enumerate(zip(outputs, batch_token_ids)):
            prompt_logprobs = output.prompt_logprobs
            
            if prompt_logprobs is None:
                all_losses.append(float('inf'))
                continue
            
            # Get mask and test length for this prompt
            mask_start = batch_mask_lengths[j] if batch_mask_lengths else 0
            mask_end = batch_test_lengths[j] if batch_test_lengths else len(token_ids)
            
            # Compute loss only for test part (from mask_start to mask_end)
            # Important: In the original implementation, shift_logits and shift_labels are used:
            #   - shift_logits = logits[..., :-1, :]  (positions 0 to n-2)
            #   - shift_labels = input_ids[..., 1:]   (positions 1 to n-1)
            #   - shift_logits[i] predicts shift_labels[i], which is input_ids[i+1]
            #   - mask is applied to shift_labels with range [mask_start, mask_end)
            #   - So shift_labels[mask_start] corresponds to input_ids[mask_start+1]
            # In vLLM, prompt_logprobs[i] contains logprobs for token at position i
            # To match the original implementation, we need to compute logprobs for positions
            # [mask_start+1, mask_end) in the original token_ids
            # But since prompt_logprobs[i] corresponds to token_ids[i], we use range [mask_start+1, mask_end)
            sum_neg_logp = 0.0
            valid_tokens = 0
            
            # Iterate over positions in the test part
            # Start from mask_start+1 to match the shift operation in original implementation
            # End at mask_end (exclusive) to match the original mask range
            for pos in range(mask_start + 1, min(mask_end, len(token_ids))):
                if pos >= len(prompt_logprobs):
                    break
                
                token_logprob_dict = prompt_logprobs[pos]
                if token_logprob_dict is None:
                    continue
                
                # Get the token ID at this position
                token_id = token_ids[pos]
                
                # Get logprob for this token
                if token_id in token_logprob_dict:
                    logprob = float(token_logprob_dict[token_id].logprob)
                    # Cross-entropy is negative log probability
                    sum_neg_logp += -logprob
                    valid_tokens += 1
            
            if valid_tokens == 0:
                all_losses.append(float('inf'))
            else:
                # Average cross-entropy loss (normalized by token length)
                all_losses.append(sum_neg_logp / valid_tokens)
    
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