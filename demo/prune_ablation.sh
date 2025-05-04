echo "Building HNSW index with 0.02per_M6-7_degree_based..."
python simple_build.py --config 0.02per_M6-7_degree_based --M 32 --efConstruction 256 --K_NEIGHBORS 3 --max_queries 1000 --batch_size 64 --db_embedding_file /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl --index_saving_dir /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices --task_name nq --embedder_model facebook/contriever-msmarco 

echo "Building HNSW index with origin..."
python simple_build.py --config origin --M 32 --efConstruction 256 --K_NEIGHBORS 3 --max_queries 1000 --batch_size 64 --db_embedding_file /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl --index_saving_dir /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices --task_name nq --embedder_model facebook/contriever-msmarco 

echo "Building HNSW index with M9_merge_edge..."
python simple_build.py --config M9_merge_edge --M 9 --efConstruction 256 --K_NEIGHBORS 3 --max_queries 1000 --batch_size 64 --db_embedding_file /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl --index_saving_dir /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices --task_name nq --embedder_model facebook/contriever-msmarco 

echo "Building HNSW index with random_delete50..."
python simple_build.py --config random_delete50 --M 32 --efConstruction 256 --K_NEIGHBORS 3 --max_queries 1000 --batch_size 64 --db_embedding_file /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl --index_saving_dir /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices --task_name nq --embedder_model facebook/contriever-msmarco 

echo "Building HNSW index with agg-1.8_merge_edge..."
python simple_build.py --config agg-1.8_merge_edge --M 32 --efConstruction 256 --K_NEIGHBORS 3 --max_queries 1000 --batch_size 64 --db_embedding_file /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl --index_saving_dir /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/indices --task_name nq --embedder_model facebook/contriever-msmarco 