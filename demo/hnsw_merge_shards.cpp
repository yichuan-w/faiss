// cd /home/zhifei/Power-RAG
// cmake --build build-faiss -j --target hnsw_merge_shards
//IDX_DIR="/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices/small_build_diskann_mips_c100_g60_s0.8_b0.8"
// build-faiss/hnsw_merge_shards \
// --out "$IDX_DIR/merged_hnsw_v2.index" \
// --R 60 --metric ip $(printf ' --shard %s' "$IDX_DIR"/hnsw_shards/*.hnsw.index)
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <chrono>

using faiss::Index;
using faiss::IndexHNSW;
using faiss::IndexIDMap;
using faiss::idx_t;

struct Shard {
    std::unique_ptr<Index> owned;              // owns the read index (IndexIDMap)
    IndexIDMap* idmap = nullptr;               // non-owning
    IndexHNSW* hnsw = nullptr;                 // sub-index (non-owning)
    std::vector<idx_t>* id_map_vec = nullptr;  // non-owning
};

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --out merged.index --R 60 --metric [ip|l2] --shard path1 --shard path2 ...\n";
}

int main(int argc, char** argv) {
    std::string out_path;
    int R = 60;
    std::string metric = "ip";
    std::vector<std::string> shard_paths;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--R") && i + 1 < argc) {
            R = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--metric") && i + 1 < argc) {
            metric = argv[++i];
        } else if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
            shard_paths.emplace_back(argv[++i]);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete arg: " << argv[i] << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (out_path.empty() || shard_paths.empty()) {
        usage(argv[0]);
        return 2;
    }

    // Load shards
    std::vector<Shard> shards;
    shards.reserve(shard_paths.size());
    int d = -1;
    faiss::MetricType mt = (metric == "l2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT);

    for (const auto& sp : shard_paths) {
        Shard s;
        s.owned.reset(faiss::read_index(sp.c_str()));
        if (!s.owned) {
            std::cerr << "Failed to read shard index: " << sp << "\n";
            return 1;
        }

        // Expect an IndexIDMap wrapping an IndexHNSW*
        s.idmap = dynamic_cast<IndexIDMap*>(s.owned.get());
        if (!s.idmap) {
            std::cerr << "Shard is not IndexIDMap: " << sp << "\n";
            return 1;
        }
        s.hnsw = dynamic_cast<IndexHNSW*>(s.idmap->index);
        if (!s.hnsw) {
            std::cerr << "Shard's sub-index is not IndexHNSW: " << sp << "\n";
            return 1;
        }
        s.id_map_vec = &s.idmap->id_map;

        if (d < 0) {
            d = s.hnsw->d;
            if (s.hnsw->metric_type != mt) {
                std::cerr << "Warning: overriding metric to shard's metric\n";
                mt = s.hnsw->metric_type;
            }
        } else {
            if (s.hnsw->d != d) {
                std::cerr << "Mismatched dimension across shards\n";
                return 1;
            }
        }
        shards.emplace_back(std::move(s));
    }

    std::cout << "Loaded " << shards.size() << " shards" << std::endl;

    // Collect global ids and a representative (shard, local) for vector reconstruction
    std::unordered_map<idx_t, std::pair<size_t, idx_t>> repr; // gid -> (shard_idx, local)
    std::unordered_set<idx_t> global_ids_set;
    for (size_t si = 0; si < shards.size(); ++si) {
        const auto& idv = *shards[si].id_map_vec;
        for (idx_t local = 0; local < (idx_t)idv.size(); ++local) {
            idx_t gid = idv[local];
            if (!global_ids_set.count(gid)) {
                global_ids_set.insert(gid);
                repr.emplace(gid, std::make_pair(si, local));
            }
        }
    }

    // Build sorted id order and mapping gid -> internal idx
    std::vector<idx_t> id_order(global_ids_set.begin(), global_ids_set.end());
    std::sort(id_order.begin(), id_order.end());
    const idx_t N = (idx_t)id_order.size();
    std::unordered_map<idx_t, idx_t> gid2int;
    gid2int.reserve(id_order.size()*2);
    for (idx_t i = 0; i < N; ++i) gid2int[id_order[i]] = i;

    std::cout << "Global unique ids: N=" << N << std::endl;

    // Build per-shard gid->local index to avoid O(n) scans
    std::vector<std::unordered_map<idx_t, idx_t>> gid2local(shards.size());
    for (size_t si = 0; si < shards.size(); ++si) {
        const auto& idv = *shards[si].id_map_vec;
        auto& m = gid2local[si];
        m.reserve(idv.size() * 2);
        for (idx_t local = 0; local < (idx_t)idv.size(); ++local) {
            m.emplace(idv[local], local);
        }
        std::cout << "Built gid->local map for shard " << si << ", size=" << idv.size() << std::endl;
    }

    // Reconstruct vectors
    std::vector<float> xb((size_t)N * (size_t)d);
    auto t0 = std::chrono::steady_clock::now();
    for (idx_t i = 0; i < N; ++i) {
        idx_t gid = id_order[i];
        auto it = repr.find(gid);
        if (it == repr.end()) {
            std::cerr << "Missing representative for gid=" << gid << "\n";
            return 1;
        }
        size_t si = it->second.first;
        idx_t local = it->second.second;
        shards[si].hnsw->reconstruct(local, xb.data() + (size_t)i * (size_t)d);
        if ((i % 100000) == 0 && i > 0) {
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "Reconstructed " << i << "/" << N << " nodes (" << (ms/1000.0) << " s)" << std::endl;
        }
    }

    // Create final index as IndexHNSWFlat so serialization is supported
    faiss::IndexHNSWFlat index(d, R, mt);
    // Add vectors to the underlying storage directly (avoid triggering HNSW build)
    index.storage->add(N, xb.data());
    index.ntotal = N;

    // Compute merged per-node maximum level across shards
    std::vector<int> merged_level_idx(N, 0); // level index (0-based); stored levels is idx+1
    int merged_max_level = 0;
    idx_t merged_entry = N > 0 ? 0 : -1;
    for (idx_t i = 0; i < N; ++i) {
        idx_t gid = id_order[i];
        int max_li = 0;
        for (size_t si = 0; si < shards.size(); ++si) {
            auto mit = gid2local[si].find(gid);
            if (mit == gid2local[si].end()) continue;
            idx_t local = mit->second;
            int local_levels = 0;
            if ((size_t)local < shards[si].hnsw->hnsw.levels.size()) {
                local_levels = shards[si].hnsw->hnsw.levels[local];
            }
            if (local_levels > 0) {
                int li = local_levels - 1; // 0-based
                if (li > max_li) max_li = li;
            }
        }
        merged_level_idx[i] = max_li;
        if (max_li > merged_max_level) {
            merged_max_level = max_li;
            merged_entry = i;
        }
    }

    // Configure HNSW topology params and allocate neighbor tables for all levels
    index.hnsw.set_default_probas(/*M=*/R, /*levelMult*/ 1.0f, /*M0*/ -1);
    index.hnsw.levels.resize(N);
    for (idx_t i = 0; i < N; ++i) index.hnsw.levels[i] = merged_level_idx[i] + 1;
    index.hnsw.offsets.clear();
    index.hnsw.offsets.push_back(0);
    index.hnsw.prepare_level_tab(N, /*preset_levels=*/true);
    index.hnsw.max_level = merged_max_level;
    index.hnsw.entry_point = merged_entry;

    // Merge neighbors for all levels (from top to bottom)
    std::mt19937 rng(123456);
    for (int level = merged_max_level; level >= 0; --level) {
        t0 = std::chrono::steady_clock::now();
        int per_level_cap = std::min(index.hnsw.nb_neighbors(level), R);
        for (idx_t i = 0; i < N; ++i) {
            if (merged_level_idx[i] < level) continue; // node not present at this level
            idx_t gid = id_order[i];
            std::unordered_set<idx_t> neigh_gid;
            // collect from all shards where this gid appears on this level
            for (size_t si = 0; si < shards.size(); ++si) {
                auto mit = gid2local[si].find(gid);
                if (mit == gid2local[si].end()) continue;
                idx_t local = mit->second;
                int local_levels = shards[si].hnsw->hnsw.levels[local];
                if (local_levels <= level) continue; // this shard doesn't have this node at this level

                size_t begin_s, end_s;
                shards[si].hnsw->hnsw.neighbor_range(local, /*level=*/level, &begin_s, &end_s);
                for (size_t j = begin_s; j < end_s; ++j) {
                    idx_t nb_local = shards[si].hnsw->hnsw.neighbors[j];
                    if (nb_local < 0) break;
                    idx_t nb_gid = shards[si].id_map_vec->at((size_t)nb_local);
                    if (nb_gid == gid) continue;
                    neigh_gid.insert(nb_gid);
                }
            }

            // shuffle and truncate to cap
            std::vector<idx_t> neigh_list;
            neigh_list.reserve(neigh_gid.size());
            for (auto g : neigh_gid) neigh_list.push_back(g);
            std::shuffle(neigh_list.begin(), neigh_list.end(), rng);
            if ((int)neigh_list.size() > per_level_cap) neigh_list.resize(per_level_cap);

            // write into neighbor array
            size_t begin, end;
            index.hnsw.neighbor_range(i, level, &begin, &end);
            size_t written = 0;
            for (; written < neigh_list.size() && begin + written < end; ++written) {
                auto git = gid2int.find(neigh_list[written]);
                if (git == gid2int.end()) continue; // skip unknown
                index.hnsw.neighbors[begin + written] = (faiss::HNSW::storage_idx_t)git->second;
            }
            // fill remainder with -1
            for (size_t j = begin + written; j < end; ++j) {
                index.hnsw.neighbors[j] = -1;
            }

            if ((i % 100000) == 0 && i > 0) {
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[L" << level << "] Merged neighbors for " << i << "/" << N << " nodes (" << (ms/1000.0) << " s)" << std::endl;
            }
        }
    }

    // Write index (without IDMap wrapper). Also dump a sidecar global ID map.
    faiss::write_index(&index, out_path.c_str());
    std::cout << "Merged index written to " << out_path << " with N=" << N << ", d=" << d << ", R=" << R << "\n";

    // Dump ids as DiskANN-style bin: uint32 n, uint32 dim(=1), followed by n uint32 ids
    std::string idmap_path = out_path + std::string("_ids_uint32.bin");
    {
        std::ofstream ofs(idmap_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open idmap for write: " << idmap_path << "\n";
        } else {
            uint32_t n32 = (uint32_t)N;
            uint32_t dim1 = 1;
            ofs.write(reinterpret_cast<const char*>(&n32), sizeof(uint32_t));
            ofs.write(reinterpret_cast<const char*>(&dim1), sizeof(uint32_t));
            for (idx_t i = 0; i < N; ++i) {
                uint32_t v = (uint32_t)id_order[(size_t)i];
                ofs.write(reinterpret_cast<const char*>(&v), sizeof(uint32_t));
            }
            ofs.close();
            std::cout << "Wrote global id map to " << idmap_path << " (" << N << " ids)\n";
        }
    }
    return 0;
}


