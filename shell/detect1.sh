# 获取pred文件夹中的所有子文件夹名
dirs= (./pred/*/)
# 遍历每个子文件夹
for dir in "$ {dirs [@]}"
do
  # 设置CUDA_VISIBLE_DEVICES为2
  export CUDA_VISIBLE_DEVICES=2
  # 运行detect.py，并将日志输出到相应的文件
  nohup python detect.py --input_dir "$dir" > ./log/detect/"$ {dir##*/}".log&
done