export CUDA_VISIBLE_DEVICES=3

# no watermark
python pred.py \
    --mode no \
    --gamma 0.5 \
    --delta 5 \
    --model llama2-7b-chat-4k