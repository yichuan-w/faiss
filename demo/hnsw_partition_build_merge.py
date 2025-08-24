# # 1) 配置并增量编译（单独的 build 目录）
# cd /home/zhifei/Power-RAG
# cmake -S faiss -B build-faiss -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Debug
# cmake --build build-faiss -j --target faiss swigfaiss hnsw_merge_shards
# uv pip install -e build-faiss/faiss/python

# # 2) 使用 DiskANN 的分片结果先为每个分片建 HNSW
# IDX_DIR="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices/small_build_diskann_mips_c100_g60_s0.8_b0.8"
# python faiss/demo/hnsw_partition_build_merge.py \
#   --base_vectors_bin "$IDX_DIR/ann_vectors.bin" \
#   --idmaps_glob "$IDX_DIR/ann_mem.index_tempFiles_subshard-*_ids_uint32.bin" \
#   --out_dir "$IDX_DIR/hnsw_shards" \
#   --R 60 --efc 200 --metric ip

import os
import sys
import glob
import json
import argparse
import struct
from typing import List, Tuple, Dict

import numpy as np


def _import_installed_faiss() -> object:
    """
    Import faiss from the installed site-packages without accidentally grabbing
    the local faiss/ source tree in this repo. Mirrors simple_build_diskann.py style.
    """
    import importlib

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    faiss_src = os.path.join(repo_root, "faiss")

    old = sys.path[:]
    try:
        # Remove repo's faiss/ from path so we import the installed bindings
        sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(repo_root)]
        sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(faiss_src)]
        mod = importlib.import_module("faiss")
        return mod
    finally:
        sys.path = old


faiss = _import_installed_faiss()


def read_idmap_uint32(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        dim = struct.unpack("<I", f.read(4))[0]
        if dim != 1:
            raise ValueError(f"idmap file {path} has dim={dim}, expected 1")
        ids = np.fromfile(f, dtype=np.uint32, count=n)
    return ids


def read_vectors_bin_float32(path: str) -> Tuple[np.ndarray, int, int]:
    """
    DiskANN-style .bin: uint32 npts, uint32 dim, then npts*dim float32
    Returns (np.ndarray [n, d], n, d)
    """
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"file too small: {path}")
        n = struct.unpack("<I", header[:4])[0]
        d = struct.unpack("<I", header[4:])[0]
        data = np.fromfile(f, dtype=np.float32, count=n * d)
    if data.size != n * d:
        raise ValueError(f"unexpected data size in {path}: got {data.size}, expected {n*d}")
    return data.reshape(n, d), n, d


def gather_rows(base_vectors: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # ids are uint32 indices into base_vectors
    return base_vectors[ids.astype(np.int64), :].copy(order="C")


def make_hnsw_index(d: int, R: int, metric: str, efc: int):
    if metric.lower() in ("ip", "inner_product", "mips"):
        quant = faiss.IndexFlatIP(d)
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        quant = faiss.IndexFlatL2(d)
        metric_type = faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(d, R, metric_type)
    # faiss python binding constructs with internal quantizer; set efConstruction
    # Some builds expose .hnsw, others use attribute on index
    if hasattr(index, "hnsw") and hasattr(index.hnsw, "efConstruction"):
        index.hnsw.efConstruction = int(efc)
    elif hasattr(index, "efConstruction"):
        index.efConstruction = int(efc)
    return index


def build_shard_hnsw(vecs: np.ndarray, ids: np.ndarray, R: int, efc: int, metric: str):
    d = vecs.shape[1]
    core = make_hnsw_index(d, R, metric, efc)
    id_map = faiss.IndexIDMap(core)
    id_map.add_with_ids(np.ascontiguousarray(vecs, dtype=np.float32), ids.astype(np.int64))
    return id_map


def save_index(index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def build_all_shards(base_vectors_path: str,
                     idmaps_glob: str,
                     out_dir: str,
                     R: int,
                     efc: int,
                     metric: str) -> Dict:
    base, n, d = read_vectors_bin_float32(base_vectors_path)
    idmap_paths = sorted(glob.glob(idmaps_glob))
    if not idmap_paths:
        raise FileNotFoundError(f"no idmaps found for pattern: {idmaps_glob}")

    shards = []
    for ipath in idmap_paths:
        # expect ..._subshard-<i>_ids_uint32.bin; extract shard id if present
        fname = os.path.basename(ipath)
        part = None
        if "_subshard-" in fname:
            try:
                part = int(fname.split("_subshard-")[1].split("_")[0])
            except Exception:
                part = None
        ids = read_idmap_uint32(ipath)
        vecs = gather_rows(base, ids)
        index = build_shard_hnsw(vecs, ids, R, efc, metric)
        shard_name = f"shard_{part}" if part is not None else os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{shard_name}.hnsw.index")
        save_index(index, out_path)
        shards.append({
            "name": shard_name,
            "idmap_path": os.path.abspath(ipath),
            "index_path": os.path.abspath(out_path),
            "num_ids": int(ids.size)
        })

    manifest = {
        "type": "hnsw_partition_shards",
        "base_vectors_bin": os.path.abspath(base_vectors_path),
        "dim": int(d),
        "metric": metric,
        "R": int(R),
        "efConstruction": int(efc),
        "shards": shards
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_merged_index_from_manifest(manifest_path: str, threaded: bool = True):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    d = int(manifest["dim"]) if "dim" in manifest else None
    is_ip = manifest.get("metric", "l2").lower() in ("ip", "inner_product", "mips")
    shards = manifest["shards"]
    # faiss.IndexShards merges multiple sub-indices logically (search-time merge), non-invasive
    merged = faiss.IndexShards(d, threaded)
    for s in shards:
        idx = faiss.read_index(s["index_path"])
        merged.add_shard(idx)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build HNSW per DiskANN partition and emit a merged loader (IndexShards)")
    parser.add_argument("--base_vectors_bin", required=True, help="DiskANN-style float32 vectors bin (n,d,data)")
    parser.add_argument("--idmaps_glob", required=True, help="Glob for *_ids_uint32.bin of partitions")
    parser.add_argument("--out_dir", required=True, help="Output directory for shard indices and manifest")
    parser.add_argument("--R", type=int, default=60, help="HNSW M (degree cap)")
    parser.add_argument("--efc", type=int, default=200, help="efConstruction")
    parser.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"], help="Similarity metric")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = build_all_shards(
        base_vectors_path=args.base_vectors_bin,
        idmaps_glob=args.idmaps_glob,
        out_dir=args.out_dir,
        R=args.R,
        efc=args.efc,
        metric=args.metric,
    )
    mpath = os.path.join(args.out_dir, "manifest.json")
    print(f"Wrote manifest: {mpath}")
    print("To load merged index at query-time:")
    print("from hnsw_partition_build_merge import load_merged_index_from_manifest as load; idx = load(\"%s\")" % mpath)


if __name__ == "__main__":
    main()


