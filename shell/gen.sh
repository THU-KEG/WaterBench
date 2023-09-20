export CUDA_VISIBLE_DEVICES=7

# old watermark
CUDA_VISIBLE_DEVICES=7 python pred.py \
    --mode old \
    --gamma 0.25 \
    --delta 2 \
    --model chatglm2-6b-32k

# v2 watermark
# python pred.py \
#     --mode v2 \
#     --gamma 0.5 \
#     --delta 5 \
#     --model llama2-7b-chat-4k

# gpt watermark
# python pred.py \
#     --mode gpt \
#     --gamma 0.5 \
#     --delta 5 \
#     --model llama2-7b-chat-4k

# new watermark
python pred.py \
    --mode new \
    --gamma 0.5 \
    --delta 5 \
    --model llama2-7b-chat-4k