export CUDA_VISIBLE_DEVICES=0

# old watermark
CUDA_VISIBLE_DEVICES=7 nohup python pred.py \
    --mode old \
    --gamma 0.75 \
    --delta 5 \
    --model llama2-7b-chat-4k > ./log/pred/pred_old_g0.75_d5.0.log&

CUDA_VISIBLE_DEVICES=6 nohup python pred.py \
    --mode old \
    --gamma 0.5 \
    --delta 5 \
    --bl_type hard \
    --model llama2-7b-chat-4k > ./log/pred/pred_old_g0.25_d2.0_hard.log&

CUDA_VISIBLE_DEVICES=4 nohup python pred.py \
    --mode old \
    --gamma 0.5 \
    --delta 5 \
    --bl_type hard \
    --model llama2-7b-chat-4k > ./log/pred/pred_old_g0.5_d5.0_hard.log&

CUDA_VISIBLE_DEVICES=0 nohup python pred.py \
    --mode old \
    --gamma 0.5 \
    --delta 2 \
    --bl_type hard \
    --model llama2-7b-chat-4k > ./log/pred/pred_old_g0.5_d2.0_hard.log&

# v2 watermark
CUDA_VISIBLE_DEVICES=4 nohup python pred.py \
    --mode v2 \
    --gamma 0.25 \
    --delta 2 \
    --model llama2-7b-chat-4k > ./log/pred/pred_v2_g0.25_d2.0.log&

# gpt watermark
CUDA_VISIBLE_DEVICES=2 nohup python pred.py \
    --mode gpt \
    --gamma 0.9 \
    --delta 5 \
    --model llama2-7b-chat-4k > ./log/pred/pred_gpt_g0.9_d5.0.log&

# new watermark
# python detect.py \
#     --input_dir ./pred/llama2-7b-chat-4k_new_g0.5_d5.0

CUDA_VISIBLE_DEVICES=7 nohup python detect.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.9_d5.0_hard > ./log/detect/detect_old_g0.9_d5.0_hard.log&
