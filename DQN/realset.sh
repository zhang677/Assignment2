worker_id=${1:-0}
echo "Current worker: ${worker_id}"

CUDA_VISIBLE_DEVICES=${worker_id} \
python main.py \
--model DQN \
--env ALE/Enduro-v5 \
--dir temp-0410-1/ \
--start 50000 --memory 1000000 --freq 10000 \
--max_eps 1 --min_eps 0.01 --dec_eps 1000000 \
--gamma 0.99 --lr 0.00025 \
--games 200 \
--leps 0.01 --m 0.95