# change whatever you need
export CUDA_VISIBLE_DEVICES=0

# old watermark
python pred.py \
    --mode old \
    --gamma 0.25 \
    --delta 5 \
    --bl_type hard \
    --model llama2-7b-chat-4k \
