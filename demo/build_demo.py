import faiss
import numpy as np
import pickle
import time
import os
import csv

# --- Configuration ---
# Make sure this path is correct for your system
EMBEDDING_FILE = "/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/passages_00.pkl"
NUM_QUERIES_FOR_EVAL = 1000  # Number of query vectors to use for evaluation
K_NEIGHBORS = 3  # Number of nearest neighbors to retrieve
# TEMP_INDEX_FILENAME = "_temp_faiss_index.bin" # No longer needed
OUTPUT_CSV_FILENAME = "index_evaluation_results.csv"
INDEX_OUTPUT_DIR = "built_faiss_indices" # Directory to save persistent indices

# HNSW Parameters
HNSW_M = 32  # Number of connections per node
EF_CONSTRUCTION_VALUES = [32, 64, 128, 256] # efConstruction values to test
EF_SEARCH_VALUES = [16, 32, 64, 128, 256]     # efSearch values to test

# NSG Parameters
NSG_R_VALUES = [16, 32, 64, 128] # R values to test for NSG build

# --- Create Output Directory ---
if not os.path.exists(INDEX_OUTPUT_DIR):
    print(f"Creating directory for indices: {INDEX_OUTPUT_DIR}")
    os.makedirs(INDEX_OUTPUT_DIR)

# --- Load Embeddings ---
def load_embeddings_pkl(filepath):
    """Loads embeddings from a pickle file, handling common structures."""
    print(f"Loading embeddings from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Handle common data structures in the pickle file
        if isinstance(data, (list, tuple)) and len(data) >= 2 and isinstance(data[1], np.ndarray):
            # Assumes structure like (ids, embeddings) - common pattern
            embeddings = data[1]
            print(f"Loaded embeddings from tuple structure (element 1), shape: {embeddings.shape}")
        elif isinstance(data, np.ndarray):
            # Assumes the pickle file directly contains the numpy array
            embeddings = data
            print(f"Loaded embeddings directly from numpy array, shape: {embeddings.shape}")
        elif isinstance(data, dict) and 'embeddings' in data and isinstance(data['embeddings'], np.ndarray):
            # Assumes dictionary structure with 'embeddings' key
            embeddings = data['embeddings']
            print(f"Loaded embeddings from dict key 'embeddings', shape: {embeddings.shape}")
        else:
            raise TypeError(f"Loaded data format not recognized or structure unexpected. Type: {type(data)}")

        # Ensure correct dtype (float32) for Faiss
        if embeddings.dtype != np.float32:
            print(f"Converting embeddings from {embeddings.dtype} to float32.")
            embeddings = embeddings.astype(np.float32)

        # Ensure C-contiguous array for Faiss performance
        if not embeddings.flags['C_CONTIGUOUS']:
            print("Embeddings array is not C-contiguous. Making a copy.")
            embeddings = np.ascontiguousarray(embeddings)

        print(f"Successfully loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        return embeddings

    except FileNotFoundError:
        print(f"Error: Embedding file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading or processing pickle file: {e}")
        exit(1)

embeddings = load_embeddings_pkl(EMBEDDING_FILE)
dim = embeddings.shape[1]
num_vectors = embeddings.shape[0]

# --- Prepare Query Set ---
if num_vectors <= NUM_QUERIES_FOR_EVAL:
    print(f"Error: Dataset size ({num_vectors}) is not larger than NUM_QUERIES_FOR_EVAL ({NUM_QUERIES_FOR_EVAL}). Cannot create separate query set.")
    exit(1)
xq_full = embeddings[-NUM_QUERIES_FOR_EVAL:]
print(f"Using last {xq_full.shape[0]} vectors from the dataset as queries for evaluation.")
print(f"All {num_vectors} vectors will be added to the indices.")

# --- Calculate Ground Truth (Flat L2) ---
print("\n--- Calculating Ground Truth (IndexFlatL2) ---")
index_flat = faiss.IndexFlatL2(dim)
start_time_build_flat = time.time()
index_flat.add(embeddings)
end_time_build_flat = time.time()
print(f"FlatL2 Add time: {end_time_build_flat - start_time_build_flat:.4f} seconds")
start_time_search_flat = time.time()
D_flat, gt_I = index_flat.search(xq_full, k=K_NEIGHBORS)
end_time_search_flat = time.time()
print(f"FlatL2 Search time: {end_time_search_flat - start_time_search_flat:.4f} seconds")
# Save ground truth index if needed (optional)
# flat_gt_path = os.path.join(INDEX_OUTPUT_DIR, "flat_ground_truth.index")
# print(f"Saving Flat ground truth index to {flat_gt_path}")
# faiss.write_index(index_flat, flat_gt_path)
del index_flat # Free memory, keep gt_I

# --- Helper Function for Recall Calculation ---
def calculate_recall(ground_truth_indices, result_indices):
    """Calculates recall@k based on ground truth."""
    num_queries = ground_truth_indices.shape[0]
    if num_queries == 0: return 0.0
    k = ground_truth_indices.shape[1]
    if k == 0: return 0.0
    recall_sum = 0
    for i in range(num_queries):
        gt_set = set(ground_truth_indices[i])
        res_set = set(result_indices[i])
        common_neighbors = len(gt_set.intersection(res_set))
        recall_sum += common_neighbors / k
    return recall_sum / num_queries

# --- Data Collection List ---
results_data = []

# --- HNSW Evaluation ---
print("\n--- Evaluating HNSW (IndexHNSWFlat) ---")
for efConstruction in EF_CONSTRUCTION_VALUES:
    print(f"\nBuilding HNSW index with M={HNSW_M}, efConstruction={efConstruction}...")
    index_hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_L2)
    index_hnsw.hnsw.efConstruction = efConstruction

    start_time_build = time.time()
    index_hnsw.add(embeddings)
    end_time_build = time.time()
    build_time_s = end_time_build - start_time_build
    print(f"Build time: {build_time_s:.4f} seconds")

    # Define persistent filename and path
    hnsw_filename = f"hnsw_M{HNSW_M}_efC{efConstruction}.index"
    hnsw_filepath = os.path.join(INDEX_OUTPUT_DIR, hnsw_filename)

    # Save index persistently and measure storage
    storage_bytes = 0
    try:
        print(f"Saving HNSW index to {hnsw_filepath}")
        faiss.write_index(index_hnsw, hnsw_filepath)
        storage_bytes = os.path.getsize(hnsw_filepath)
        # os.remove(TEMP_INDEX_FILENAME) # No longer remove
        print(f"Index storage size: {storage_bytes / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Warning: Could not save or measure index size. Error: {e}")

    approx_net_storage = num_vectors * HNSW_M # Simplified metric

    # Search loop
    for efSearch in EF_SEARCH_VALUES:
        print(f"  Searching HNSW with efSearch={efSearch}...")
        index_hnsw.hnsw.efSearch = efSearch
        start_time_search = time.time()
        D_hnsw, I_hnsw = index_hnsw.search(xq_full, K_NEIGHBORS)
        end_time_search = time.time()
        search_time_s = end_time_search - start_time_search
        print(f"  Search time: {search_time_s:.4f} seconds")

        recall = calculate_recall(gt_I, I_hnsw)
        print(f"  Recall@{K_NEIGHBORS}: {recall:.4f}")

        # Record results
        results_data.append({
            "index_type": "HNSWFlat",
            "build_param_name": "efConstruction",
            "build_param_value": efConstruction,
            "search_param_name": "efSearch",
            "search_param_value": efSearch,
            "M_or_R": HNSW_M,
            "recall": recall,
            "storage_bytes": storage_bytes,
            "build_time_s": build_time_s,
            "search_time_s": search_time_s,
            "approx_net_storage": approx_net_storage,
            "index_filename": hnsw_filename # Add filename to results
        })

    del index_hnsw # Free memory

# --- NSG Evaluation ---
print("\n--- Evaluating NSG (IndexNSGFlat) ---")
for nsg_r_val in NSG_R_VALUES:
    print(f"\nBuilding NSG index with R={nsg_r_val}...")
    quantizer_nsg = faiss.IndexFlatL2(dim)
    index_nsg = faiss.IndexNSGFlat(quantizer_nsg, dim, nsg_r_val, faiss.METRIC_L2)

    start_time_build = time.time()
    index_nsg.add(embeddings)
    end_time_build = time.time()
    build_time_s = end_time_build - start_time_build
    print(f"Build time: {build_time_s:.4f} seconds")

    # Define persistent filename and path
    nsg_filename = f"nsg_R{nsg_r_val}.index"
    nsg_filepath = os.path.join(INDEX_OUTPUT_DIR, nsg_filename)

    # Save index persistently and measure storage
    storage_bytes = 0
    try:
        print(f"Saving NSG index to {nsg_filepath}")
        faiss.write_index(index_nsg, nsg_filepath)
        storage_bytes = os.path.getsize(nsg_filepath)
        # os.remove(TEMP_INDEX_FILENAME) # No longer remove
        print(f"Index storage size: {storage_bytes / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Warning: Could not save or measure index size. Error: {e}")

    approx_net_storage = num_vectors * nsg_r_val # Simplified metric

    print(f"  Searching NSG (R={nsg_r_val})...")
    start_time_search = time.time()
    D_nsg, I_nsg = index_nsg.search(xq_full, K_NEIGHBORS)
    end_time_search = time.time()
    search_time_s = end_time_search - start_time_search
    print(f"  Search time: {search_time_s:.4f} seconds")

    recall = calculate_recall(gt_I, I_nsg)
    print(f"  Recall@{K_NEIGHBORS}: {recall:.4f}")

    # Record results
    results_data.append({
        "index_type": "NSGFlat",
        "build_param_name": "R",
        "build_param_value": nsg_r_val,
        "search_param_name": "N/A",
        "search_param_value": None,
        "M_or_R": nsg_r_val,
        "recall": recall,
        "storage_bytes": storage_bytes,
        "build_time_s": build_time_s,
        "search_time_s": search_time_s,
        "approx_net_storage": approx_net_storage,
        "index_filename": nsg_filename # Add filename to results
    })

    del index_nsg # Free memory

# --- Write Results to CSV ---
if results_data:
    print(f"\nWriting evaluation results to {OUTPUT_CSV_FILENAME}...")
    fieldnames = list(results_data[0].keys()) # Ensure field order consistency
    try:
        with open(OUTPUT_CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        print("Successfully wrote results to CSV.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
else:
    print("No results were collected to write to CSV.")

print("\nEvaluation complete.")
