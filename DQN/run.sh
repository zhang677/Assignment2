worker_id=${1}
dir=${2}
echo "Current worker: ${worker_id}"

CUDA_VISIBLE_DEVICES=${worker_id} \
python main.py \
--model DQN \
--env ALE/Enduro-v5 \
--dir ${dir} \
--start 320 --memory 10000 --freq 1000 \
--max_eps 1 --min_eps 0.01 --dec_eps 99000 \
--gamma 0.99 --lr 0.0001 \
--games 200 \
--leps 1e-08 --m 0
