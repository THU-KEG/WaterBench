# no watermark
CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_no_g0.5_d5.0 > ./log/eval/detect_no_g0.5_d5.0.log&

# gpt_g0.1_d10.0
CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.1_d10.0 > ./log/eval/detect_gpt_g0.1_d10.0.log&


# gpt_g0.25_d15.0
CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.25_d15.0 > ./log/eval/detect_gpt_g0.25_d15.0.log&

# gpt_g0.25_d10.0
CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.5_d10.0 > ./log/eval/detect_gpt_g0.25_d10.0.log&

# old_g0.1_d10.0
CUDA_VISIBLE_DEVICES=3 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.1_d10.0 > ./log/eval/detect_old_g0.1_d10.0.log&

# old_g0.5_d2.0_hard
CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.5_d2.0_hard > ./log/eval/detect_old_g0.5_d2.0_hard.log&

# old_g0.25_d2.0
CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.25_d2.0 > ./log/eval/detect_old_g0.25_d2.0.log&

# old_g0.25_d5.0_hard
CUDA_VISIBLE_DEVICES=3 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.25_d5.0_hard > ./log/eval/detect_old_g0.25_d5.0_hard.log&

# old_g0.25_d15.0
CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.25_d15.0 > ./log/eval/detect_old_g0.25_d15.0.log&

# v2_g0.25_d15.0
CUDA_VISIBLE_DEVICES=5 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_v2_g0.25_d15.0 > ./log/eval/detect_v2_g0.25_d15.0.log&

# v2_g0.25_d10.0
CUDA_VISIBLE_DEVICES=6 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_v2_g0.25_d10.0 > ./log/eval/detect_v2_g0.25_d10.0.log&

nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.75_d15.0 > ./log/eval/detect_old_g0.75_d15.0.log&

nohup python eval.py \
  --input_dir ./pred/llama2-7b-chat-4k_old_g0.75_d5.0_hard > ./log/eval/detect_old_g0.75_d5.0_hard.log&
  
nohup python eval.py \
  --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.65_d12.5 > ./log/eval/detect_gpt_g0.65_d12.5.log&

nohup python eval.py \
--input_dir ./pred/llama2-7b-chat-4k_v2_g0.75_d15.0 > ./log/eval/detect_v2_g0.75_d15.0.log&
  
# internlm-7b-8k
# no watermark
CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
    --input_dir ./pred/internlm-7b-8k_no_g0.1_d10.0 > ./log/eval/internlm-7b-8k/detect_no_g0.1_d10.0.log&

# v2_g0.1_d10.0
CUDA_VISIBLE_DEVICES=6 nohup python eval.py \
    --input_dir ./pred/internlm-7b-8k_v2_g0.1_d10.0 > ./log/eval/internlm-7b-8k/detect_v2_g0.1_d10.0.log&

# old_g0.15_d10.0_hard
CUDA_VISIBLE_DEVICES=7 nohup python eval.py \
    --input_dir ./pred/internlm-7b-8k_old_g0.15_d10.0_hard > ./log/eval/internlm-7b-8k/detect_old_g0.15_d10.0_hard.log&

# gpt_g0.25_d15.0
CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
    --input_dir ./pred/internlm-7b-8k_gpt_g0.25_d15.0 > ./log/eval/internlm-7b-8k/detect_gpt_g0.25_d15.0.log&

# old_g0.1_d10.0
CUDA_VISIBLE_DEVICES=5 nohup python eval.py \
    --input_dir ./pred/internlm-7b-8k_old_g0.1_d10.0 > ./log/eval/internlm-7b-8k/detect_old_g0.1_d10.0.log&

