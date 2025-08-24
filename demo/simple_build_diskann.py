import sys
import time
import numpy as np
import pickle
import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Optional heavy deps imported lazily

parser = argparse.ArgumentParser(description="Build and evaluate DiskANN index (MIPS)")
parser.add_argument('--config', type=str, default="small_build", help='Configuration name for the index subdir')
parser.add_argument('--K_NEIGHBORS', type=int, default=3, help='Number of neighbors to retrieve (default: 3)')
parser.add_argument('--max_queries', type=int, default=1000, help='Maximum number of queries to load (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for query encoding (default: 64)')
parser.add_argument('--db_embedding_file', type=str, default="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl", help='Path to database embedding file')
parser.add_argument('--index_saving_dir', type=str, default="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices", help='Directory to save the index')
parser.add_argument('--task_name', type=str, default="nq", help='Task name from TASK_CONFIGS (default: nq)')
parser.add_argument('--embedder_model', type=str, default="facebook/contriever-msmarco", help='Model name for query embedding (default: facebook/contriever-msmarco)')

# DiskANN-specific
parser.add_argument('--complexity', type=int, default=100, help='DiskANN build/search complexity (default: 100)')
parser.add_argument('--graph_degree', type=int, default=60, help='DiskANN graph degree R (default: 60)')
parser.add_argument('--search_memory_maximum', type=float, default=60, help='DiskANN search memory maximum in GB (default: 60)')
parser.add_argument('--build_memory_maximum', type=float, default=350, help='DiskANN build memory maximum in GB (default: 350)')
parser.add_argument('--num_threads', type=int, default=64, help='Number of threads for DiskANN (default: 64)')
parser.add_argument('--beam_width', type=int, default=1, help='DiskANN beam width for search (default: 1)')
parser.add_argument('--nodes_to_cache', type=int, default=10000, help='Number of nodes to cache when loading DiskANN (default: 10000)')

# Search toggles
parser.add_argument('--use_deferred_fetch', action='store_true', help='Enable deferred fetch in search')
parser.add_argument('--skip_search_reorder', action='store_true', help='Skip search reorder')
parser.add_argument('--dedup_node_dis', action='store_true', help='Deduplicate node distances')

args = parser.parse_args()

K_NEIGHBORS = args.K_NEIGHBORS
DB_EMBEDDING_FILE = args.db_embedding_file
INDEX_SAVING_DIR = args.index_saving_dir
TASK_NAME = args.task_name
EMBEDDER_MODEL_NAME = args.embedder_model
MAX_QUERIES_TO_LOAD = args.max_queries
QUERY_ENCODING_BATCH_SIZE = args.batch_size
CONFIG_NAME = args.config

COMPLEXITY = args.complexity
GRAPH_DEGREE = args.graph_degree
SEARCH_MEM_GB = args.search_memory_maximum
BUILD_MEM_GB = args.build_memory_maximum
NUM_THREADS = args.num_threads
BEAM_WIDTH = args.beam_width
NODES_TO_CACHE = args.nodes_to_cache

USE_DEFERRED_FETCH = args.use_deferred_fetch
SKIP_SEARCH_REORDER = args.skip_search_reorder
DEDUP_NODE_DIS = args.dedup_node_dis

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "demo"))
from config import TASK_CONFIGS
sys.path.append(project_root)
from contriever.src.contriever import load_retriever

print(f"Loading embeddings from {DB_EMBEDDING_FILE}...")
with open(DB_EMBEDDING_FILE, 'rb') as f:
    data = pickle.load(f)

xb = data[1]
print(f"Original dtype: {xb.dtype}")
if xb.dtype != np.float32:
    print("Converting embeddings to float32.")
    xb = xb.astype(np.float32)
print(f"Loaded database embeddings (xb), shape: {xb.shape}")
d = xb.shape[1]

query_file_path = TASK_CONFIGS[TASK_NAME].query_path
print(f"Using query path from TASK_CONFIGS: {query_file_path}")

query_texts = []
print(f"Reading queries from: {query_file_path}")
with open(query_file_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= MAX_QUERIES_TO_LOAD:
            print(f"Stopped loading queries at limit: {MAX_QUERIES_TO_LOAD}")
            break
        record = json.loads(line)
        query_texts.append(record["query"])
print(f"Loaded {len(query_texts)} query texts.")

print("\nInitializing retriever model for encoding queries...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, tokenizer, _ = load_retriever(EMBEDDER_MODEL_NAME)
model.to(device)
model.eval()
print("Retriever model loaded.")


def embed_queries(queries, model, tokenizer, model_name_or_path, per_gpu_batch_size=64):
    model = model.half()
    model.eval()
    embeddings = []
    batch_question = []

    with torch.no_grad():
        for k, query in tqdm(enumerate(queries), desc="Encoding queries"):
            batch_question.append(query)
            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                if "contriever" not in model_name_or_path:
                    output = output.last_hidden_state[:, 0, :]
                embeddings.append(output.cpu())
                batch_question = []

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Query embeddings shape: {embeddings.shape}")
    return embeddings


print(f"\nEncoding {len(query_texts)} queries (batch size: {QUERY_ENCODING_BATCH_SIZE})...")
xq_full = embed_queries(query_texts, model, tokenizer, EMBEDDER_MODEL_NAME, per_gpu_batch_size=QUERY_ENCODING_BATCH_SIZE)
if xq_full.dtype != np.float32:
    print(f"Converting encoded queries from {xq_full.dtype} to float32.")
    xq_full = xq_full.astype(np.float32)
print(f"Encoded queries (xq_full), shape: {xq_full.shape}, dtype: {xq_full.dtype}")

if xq_full.shape[1] != d:
    raise ValueError(f"Query embedding dimension ({xq_full.shape[1]}) does not match database dimension ({d})")

# Ground truth via FlatIP — ensure we import installed FAISS, not local source tree
import importlib

def _import_installed_faiss() -> object:
    old = list(sys.path)
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        local_faiss_dir = os.path.join(repo_root, "faiss")
        # Build filtered sys.path temporarily
        filtered = []
        for p in old:
            ap = os.path.abspath(p)
            # Drop repo root and its local faiss directory
            if ap == repo_root or ap == local_faiss_dir:
                continue
            # Drop any path that directly exposes a top-level 'faiss' folder (non site-packages)
            try:
                has_local = os.path.isdir(os.path.join(ap, "faiss"))
            except Exception:
                has_local = False
            if has_local and ("site-packages" not in ap and "dist-packages" not in ap):
                continue
            filtered.append(p)
        sys.path = filtered
        if "faiss" in sys.modules:
            del sys.modules["faiss"]
        mod = importlib.import_module("faiss")
        return mod
    finally:
        # Restore sys.path regardless of import success
        sys.path = old

faiss = _import_installed_faiss()
print("\nBuilding FlatIP index for ground truth...")
# Prefer IndexFlatIP if available; otherwise use index factory as a fallback
if hasattr(faiss, "IndexFlatIP"):
    index_flat = faiss.IndexFlatIP(d)
elif hasattr(faiss, "IndexFlat") and hasattr(faiss, "METRIC_INNER_PRODUCT"):
    index_flat = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
else:
    # Fallback to factory string
    if hasattr(faiss, "index_factory"):
        index_flat = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    else:
        raise RuntimeError("FAISS Python bindings missing FlatIP/Flat or index_factory")
index_flat.add(xb)
print(f"Searching FlatIP index with {MAX_QUERIES_TO_LOAD} queries (k={K_NEIGHBORS})...")
_, recall_idx_flat = index_flat.search(xq_full, k=K_NEIGHBORS)

# Prepare DiskANN index dir
index_dir = f"{INDEX_SAVING_DIR}/{CONFIG_NAME}_diskann_mips_c{COMPLEXITY}_g{GRAPH_DEGREE}_s{SEARCH_MEM_GB}_b{BUILD_MEM_GB}"
os.makedirs(index_dir, exist_ok=True)


def build_or_load_diskann(xb: np.ndarray, index_dir: str):
    import diskannpy
    # Try to load if index files present
    existing_files = list(Path(index_dir).glob("ann*"))
    if len(existing_files) > 0:
        print(f"Found existing DiskANN index at {index_dir}, loading...")
        index = diskannpy.StaticDiskIndex(
            index_directory=index_dir,
            num_threads=NUM_THREADS,
            num_nodes_to_cache=NODES_TO_CACHE,
            cache_mechanism=0,
            distance_metric="mips",
            vector_dtype=np.float32,
            dimensions=d,
            index_prefix="ann",
        )
        print("DiskANN index loaded successfully.")
        return index

    print('Building DiskANN index (MIPS)...')
    import diskannpy
    from diskannpy._files import vectors_to_file  # type: ignore
    start_time = time.time()
    # Write vectors to expected bin path for this vendored builder
    vector_bin_path = os.path.join(index_dir, "ann_vectors.bin")
    if not os.path.exists(vector_bin_path):
        arr = np.ascontiguousarray(xb.astype(np.float32))
        vectors_to_file(vector_file=vector_bin_path, vectors=arr)
    # Prepare max_base_norm file required by vendored MIPS build path
    base_norm_path = os.path.join(index_dir, "ann_disk.index_max_base_norm.bin")
    try:
        # Compute max L2 norm across base vectors
        max_norm = float(np.max(np.linalg.norm(xb.astype(np.float32), axis=1)))
        with open(base_norm_path, 'wb') as fout:
            np.array([1], dtype=np.uint64).tofile(fout)  # npts
            np.array([1], dtype=np.uint64).tofile(fout)  # dims
            np.array([max_norm], dtype=np.float32).tofile(fout)
    except Exception as e:
        print(f"Warning: failed to write max_base_norm file at {base_norm_path}: {e}")
    diskannpy.build_disk_index(
        data=vector_bin_path,
        distance_metric="mips",
        index_directory=index_dir,
        complexity=COMPLEXITY,
        graph_degree=GRAPH_DEGREE,
        search_memory_maximum=SEARCH_MEM_GB,
        build_memory_maximum=BUILD_MEM_GB,
        num_threads=NUM_THREADS,
        pq_disk_bytes=0,
        vector_dtype=np.float32,
        index_prefix="ann",
    )
    end_time = time.time()
    print(f"DiskANN build time: {end_time - start_time:.3f}s")

    index = diskannpy.StaticDiskIndex(
        index_directory=index_dir,
        num_threads=NUM_THREADS,
        num_nodes_to_cache=NODES_TO_CACHE,
        cache_mechanism=0,
        distance_metric="mips",
        vector_dtype=np.float32,
        dimensions=d,
        index_prefix="ann",
    )
    return index


index = build_or_load_diskann(xb, index_dir)

print('\nSearching DiskANN index...')
recall_result_file = f"{index_dir}/recall_result.txt"
time_list = []
recall_list = []

# Sweep complexities for search evaluation
complexity_list = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
with open(recall_result_file, 'w') as f:
    for complexity in complexity_list:
        start_time = time.time()
        results = index.batch_search(
            xq_full,
            k_neighbors=K_NEIGHBORS,
            complexity=complexity,
            num_threads=NUM_THREADS,
            beam_width=BEAM_WIDTH,
            USE_DEFERRED_FETCH=USE_DEFERRED_FETCH,
            skip_search_reorder=SKIP_SEARCH_REORDER,
            dedup_node_dis=DEDUP_NODE_DIS,
        )
        end_time = time.time()
        D = results.distances
        I = results.identifiers
        print('D[0]:', D[0])
        elapsed = end_time - start_time
        print(f'time: {elapsed:.6f}s')
        time_list.append(elapsed)

        # Recall@K vs flat
        recall = []
        for i in range(len(I)):
            acc = 0
            for j in range(len(I[i])):
                if I[i][j] in recall_idx_flat[i]:
                    acc += 1
            recall.append(acc / len(I[i]))
        recall = float(sum(recall) / len(recall)) if len(recall) > 0 else 0.0
        recall_list.append(recall)
        print(f'complexity: {complexity}')
        print(f'recall: {recall}')
        f.write(f'complexity: {complexity}, recall: {recall}\n')

print(f'Done and result saved to {recall_result_file}')
print(f'time_list: {time_list}')
print(f'recall_list: {recall_list}')

