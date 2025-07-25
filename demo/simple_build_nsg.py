import sys
import time
import faiss
import numpy as np
import pickle
import os
import json
import time
import torch
from tqdm import tqdm
from pathlib import Path
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "demo"))
from config import SCALING_OUT_DIR, get_example_path, TASK_CONFIGS, get_embedding_path
sys.path.append(project_root)
from contriever.src.contriever import Contriever, load_retriever

M = 20
efConstruction = 256
K_NEIGHBORS = 3

# New configuration using DPR
DOMAIN_NAME = "rpj_wiki"
EMBEDDER_NAME = "facebook/contriever-msmarco"
TASK_NAME = "nq"
MAX_QUERIES_TO_LOAD = 1000
QUERY_ENCODING_BATCH_SIZE = 64

# Get the embedding path using the function from config
# embed_path = get_embedding_path(DOMAIN_NAME, EMBEDDER_NAME, 0)
# INDEX_SAVING_FILE = os.path.join(os.path.dirname(embed_path), "indices")
# os.makedirs(INDEX_SAVING_FILE, exist_ok=True)

# Original configuration (commented out)
embed_path = "/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl"
INDEX_SAVING_FILE = "/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices"

# Load embeddings
print(f"Loading embeddings from {extend_path}...")
with open(embed_path, 'rb') as f:
    data = pickle.load(f)

xb = data[1]
print(f"Original dtype: {xb.dtype}")

if xb.dtype != np.float32:
    print("Converting embeddings to float32.")
    xb = xb.astype(np.float32)
else:
    print("Embeddings are already float32.")
print(f"Loaded database embeddings (xb), shape: {xb.shape}")
d = xb.shape[1] # Get dimension

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
model, tokenizer, _ = load_retriever(EMBEDDER_NAME)
model.to(device)
model.eval() # Set to evaluation mode
print("Retriever model loaded.")


def embed_queries(queries, model, tokenizer, model_name_or_path, per_gpu_batch_size=64):
    """Embed queries using the model with batching"""
    model = model.half()
    model.eval()
    embeddings = []
    batch_question = []

    with torch.no_grad():
        for k, query in tqdm(enumerate(queries), desc="Encoding queries"):
            batch_question.append(query)

            # Process when batch is full or at the end
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

                # Contriever typically uses output.last_hidden_state pooling or something specialized
                if "contriever" not in model_name_or_path:
                    output = output.last_hidden_state[:, 0, :]

                embeddings.append(output.cpu())
                batch_question = []  # Reset batch

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Query embeddings shape: {embeddings.shape}")
    return embeddings

print(f"\nEncoding {len(query_texts)} queries (batch size: {QUERY_ENCODING_BATCH_SIZE})...")
xq_full = embed_queries(query_texts, model, tokenizer, EMBEDDER_NAME, per_gpu_batch_size=QUERY_ENCODING_BATCH_SIZE)

# Ensure float32 for Faiss compatibility after encoding
if xq_full.dtype != np.float32:
    print(f"Converting encoded queries from {xq_full.dtype} to float32.")
    xq_full = xq_full.astype(np.float32)

print(f"Encoded queries (xq_full), shape: {xq_full.shape}, dtype: {xq_full.dtype}")

# Check dimension consistency
if xq_full.shape[1] != d:
     raise ValueError(f"Query embedding dimension ({xq_full.shape[1]}) does not match database dimension ({d})")

# Build flat index for ground truth
print("\nBuilding FlatIP index for ground truth...")
index_flat = faiss.IndexFlatIP(d)  # Use Inner Product
index_flat.add(xb)
print(f"Searching FlatIP index with {len(xq_full)} queries (k={K_NEIGHBORS})...")
D_flat, recall_idx_flat = index_flat.search(xq_full, k=K_NEIGHBORS)

# Create a specific directory for this index configuration
index_dir = f"{INDEX_SAVING_FILE}/rpj_wiki_nsg_IP_M{M}"
os.makedirs(index_dir, exist_ok=True)
index_filename = f"{index_dir}/index.faiss"

# Check if index already exists
if os.path.exists(index_filename):
    print(f"Found existing index at {index_filename}, loading...")
    index = faiss.read_index(index_filename)
    print("Index loaded successfully.")
else:
    print('Building HNSW index (IP)...')
    # add build time
    start_time = time.time()
    index = faiss.IndexNSGFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index.add(xb)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    print('HNSW index built.')
    
    # Save the HNSW index
    print(f"Saving index to {index_filename}...")
    faiss.write_index(index, index_filename)
    print("Index saved successfully.")

# Analyze the HNSW index
print("\nAnalyzing HNSW index...")
print(f"Total number of nodes: {index.ntotal}")
print("Neighbor statistics:")
print(index.nsg.print_neighbor_stats(0))

# Save degree distribution
distribution_filename = f"{index_dir}/degree_distribution.txt"
print(f"Saving degree distribution to {distribution_filename}...")
index.nsg.save_degree_distribution(distribution_filename)
print("Degree distribution saved successfully.")

# Plot the degree distribution
plot_output_path = f"{index_dir}/degree_distribution.png"
print(f"Generating degree distribution plot to {plot_output_path}...")
try:
    subprocess.run(
        ["python", f"{project_root}/utils/plot_degree_distribution.py", distribution_filename, "-o", plot_output_path],
        check=True
    )
    print(f"Degree distribution plot saved to {plot_output_path}")
except subprocess.CalledProcessError as e:
    print(f"Error generating degree distribution plot: {e}")
except FileNotFoundError:
    print("Warning: plot_degree_distribution.py script not found in specified path")

print('Searching HNSW index...')

recall_result_file = f"{index_dir}/recall_result.txt"
with open(recall_result_file, 'w') as f:
    for efSearch in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        index.nsg.efSearch = efSearch
        # calculate the time of searching
        start_time = time.time()
        
        D, I = index.search(xq_full, K_NEIGHBORS)
        end_time = time.time()
        print(f'time: {end_time - start_time}')

        # calculate the recall using the flat index
        recall = []
        for i in range(len(I)):
            acc = 0
            for j in range(len(I[i])):
                if I[i][j] in recall_idx_flat[i]:
                    acc += 1
            recall.append(acc / len(I[i]))
        recall = sum(recall) / len(recall)
        print(f'efSearch: {efSearch}')
        print(f'recall: {recall}')
        f.write(f'efSearch: {efSearch}, recall: {recall}\n')
print(f'Done and result saved to {recall_result_file}')