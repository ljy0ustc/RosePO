CUDA_VISIBLE_DEVICES=0 python evaluate_reranking.py --dataset movielens --llm_path ./ckpt/movielens/RosePO --batch_size 16 --seed 0
CUDA_VISIBLE_DEVICES=0 python evaluate_reranking.py --dataset movielens --llm_path ./ckpt/movielens/SFT --batch_size 16 --seed 0

CUDA_VISIBLE_DEVICES=0 python evaluate_reranking.py --dataset goodreads --llm_path ./ckpt/goodreads/RosePO --batch_size 8 --seed 0
CUDA_VISIBLE_DEVICES=0 python evaluate_reranking.py --dataset goodreads --llm_path ./ckpt/goodreads/SFT --batch_size 8 --seed 0

CUDA_VISIBLE_DEVICES=0 python evaluate_reranking.py --dataset steam --llm_path ./ckpt/steam/RosePO --batch_size 16 --seed 0
CUDA_VISIBLE_DEVICES=4 python evaluate_reranking.py --dataset steam --llm_path ./ckpt/steam/SFT --batch_size 16 --seed 4