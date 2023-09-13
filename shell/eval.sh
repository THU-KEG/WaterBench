# no watermark
CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_no_g0.5_d5.0 > ./log/eval/detect_no_g0.5_d5.0.log&