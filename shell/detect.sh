export CUDA_VISIBLE_DEVICES=0

ps -aux |grep -v grep|grep PID_NUM

# no watermark
CUDA_VISIBLE_DEVICES=0 nohup python pred.py \
    --mode no \
    --gamma 0.1 \
    --delta 10 \
    --model internlm-7b-8k > ./log/pred/internlm-7b-8k/pred_no_g0.1_d10.0.log&

# old watermark
CUDA_VISIBLE_DEVICES=3 nohup python pred.py \
    --mode old \
    --gamma 0.1 \
    --delta 10 \
    --dataset alpacafarm \
    --model internlm-7b-8k >> ./log/pred/internlm-7b-8k/pred_old_g0.1_d10.0_alpacafarm.log&

CUDA_VISIBLE_DEVICES=5 nohup python pred.py \
    --mode old \
    --gamma 0.15 \
    --delta 10 \
    --bl_type hard \
    --dataset alpacafarm \
    --model internlm-7b-8k >> ./log/pred/internlm-7b-8k/pred_old_g0.15_d10.0_hard_alpacafarm.log&

CUDA_VISIBLE_DEVICES=7 nohup python pred.py \
    --mode old \
    --gamma 0.75 \
    --delta 2 \
    --model llama2-7b-chat-4k > ./log/pred/llama2-7b-chat-4k/pred_old_g0.75_d2.0.log&

CUDA_VISIBLE_DEVICES=0 nohup python pred.py \
    --mode old \
    --gamma 0.5 \
    --delta 2 \
    --bl_type hard \
    --model llama2-7b-chat-4k > ./log/pred/pred_old_g0.5_d2.0_hard.log&

# v2 watermark
CUDA_VISIBLE_DEVICES=3 nohup python pred.py \
    --mode v2 \
    --gamma 0.9 \
    --delta 2 \
    --model llama2-7b-chat-4k > ./log/pred/llama2-7b-chat-4k/pred_v2_g0.9_d2.0.log&

# gpt watermark
CUDA_VISIBLE_DEVICES=4 nohup python pred.py \
    --mode gpt \
    --gamma 0.9 \
    --delta 2 \
    --model llama2-7b-chat-4k > ./log/pred/llama2-7b-chat-4k/pred_gpt_g0.9_d2.0.log&

# new watermark
# python detect.py \
#     --input_dir ./pred/llama2-7b-chat-4k_new_g0.5_d5.0

CUDA_VISIBLE_DEVICES=7 nohup python detect.py \
    --input_dir ./pred/llama2-7b-chat-4k_v2_g0.75_d15.0 > ./log/detect/llama2-7b-chat-4k/detect_v2_g0.75_d15.0.log&

CUDA_VISIBLE_DEVICES=5 nohup python detect.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.75_d2.0 > ./log/detect/llama2-7b-chat-4k/detect_old_g0.75_d2.0.log&


CUDA_VISIBLE_DEVICES=5 nohup python detect.py \
    --input_dir ./pred/internlm-7b-8k_old_g0.1_d2.0 > ./log/detect/internlm-7b-8k/detect_old_g0.1_d10.0.log&

python -m process.process_z

CUDA_VISIBLE_DEVICES=2 nohup python mutual_detect.py \
    --reference_dir internlm-7b-8k_old_g0.15_d10.0_hard \
    --detect_dir internlm-7b-8k_no_g0.1_d10.0 > ./log/mutual_detect/internlm-7b-8k_old_g0.15_d10.0_hard/no_g0.1_d10.0.log&

CUDA_VISIBLE_DEVICES=3 nohup python mutual_detect.py \
    --reference_dir internlm-7b-8k_old_g0.15_d10.0_hard \
    --detect_dir human_generation > ./log/mutual_detect/internlm-7b-8k_old_g0.15_d10.0_hard/human_generation.log&

CUDA_VISIBLE_DEVICES=4 nohup python mutual_detect.py \
    --reference_dir internlm-7b-8k_old_g0.15_d10.0_hard \
    --detect_dir internlm-7b-8k_gpt_g0.25_d15.0 > ./log/mutual_detect/internlm-7b-8k_old_g0.15_d10.0_hard/gpt_g0.25_d15.0.log&

CUDA_VISIBLE_DEVICES=5 nohup python mutual_detect.py \
    --reference_dir internlm-7b-8k_old_g0.15_d10.0_hard \
    --detect_dir internlm-7b-8k_v2_g0.1_d10.0 > ./log/mutual_detect/internlm-7b-8k_old_g0.15_d10.0_hard/v2_g0.1_d10.0.log&

CUDA_VISIBLE_DEVICES=6 nohup python mutual_detect.py \
    --reference_dir internlm-7b-8k_old_g0.15_d10.0_hard \
    --detect_dir internlm-7b-8k_old_g0.1_d10.0 > ./log/mutual_detect/internlm-7b-8k_old_g0.15_d10.0_hard/old_g0.1_d10.0.log&

CUDA_VISIBLE_DEVICES=3 nohup python mutual_detect.py \
    --reference_dir llama2-7b-chat-4k_gpt_g0.1_d10.0 \
    --detect_dir human_generation > ./log/mutual_detect/llama2-7b-chat-4k_gpt_g0.1_d10.0/human_generation.log&

CUDA_VISIBLE_DEVICES=2 nohup python mutual_detect.py \
    --reference_dir llama2-7b-chat-4k_old_g0.1_d10.0 \
    --detect_dir human_generation > ./log/mutual_detect/llama2-7b-chat-4k_old_g0.1_d10.0/human_generation.log&

CUDA_VISIBLE_DEVICES=1 nohup python mutual_detect.py \
    --reference_dir llama2-7b-chat-4k_old_g0.5_d2.0_hard \
    --detect_dir human_generation > ./log/mutual_detect/llama2-7b-chat-4k_old_g0.5_d2.0_hard/human_generation.log&

CUDA_VISIBLE_DEVICES=7 nohup python mutual_detect.py \
    --reference_dir llama2-7b-chat-4k_old_g0.25_d5.0_hard \
    --detect_dir human_generation > ./log/mutual_detect/llama2-7b-chat-4k_old_g0.25_d5.0_hard/human_generation.log&

38,llama2-7b-chat-4k,v2,0.25,15.0,4,None,22.373182046234906,0.9413477537437605,0.058652246256239604,2404

40,llama2-7b-chat-4k,old,0.1,10.0,4,soft,37.2340361891491,0.9488352745424293,0.05116472545757072,2404

6,llama2-7b-chat-4k,old,0.25,5.0,4,hard,22.937678243910696,0.9530145530145531,0.04698544698544699,2405

1,llama2-7b-chat-4k,gpt,0.1,10.0,4,None,39.75813679567962,0.9667359667359667,0.033264033264033266,2405

25,llama2-7b-chat-4k,old,0.25,15.0,4,soft,22.992148659085533,0.9667359667359667,0.033264033264033266,2405

2,llama2-7b-chat-4k,old,0.25,15.0,4,hard,23.099709968804817,0.9725571725571726,0.027442827442827444,2405



5,llama2-7b-chat-4k,old,0.5,2.0,4,hard,11.441775173258632,0.8361746361746362,0.16382536382536383,2405

21,llama2-7b-chat-4k,gpt,0.25,15.0,4,None,21.09114299144927,0.8657774072530221,0.1342225927469779,2399

9,llama2-7b-chat-4k,v2,0.25,10.0,4,None,20.745836135007735,0.8752598752598753,0.12474012474012475,2405

16,llama2-7b-chat-4k,old,0.25,2.0,4,soft,14.551659752574135,0.8864864864864865,0.11351351351351352,2405

12,internlm-7b-8k,v2,0.1,10.0,4,None,20.651504874074814,0.9511456982274102,0.04885430177258971,2313
14,internlm-7b-8k,old,0.15,10.0,4,hard,17.036902333801216,0.9547038327526133,0.04529616724738676,2296
6,internlm-7b-8k,gpt,0.25,15.0,4,None,23.56013772244894,0.9677829588808817,0.03221704111911827,2359
9,internlm-7b-8k,old,0.1,10.0,4,soft,19.758862891318824,0.969217238346526,0.030782761653474055,2274