# 导入transformers库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import random
import json
from generate import Generator
from argparse import Namespace


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
seed_everything(42)
model2path = json.load(open("config/model2path.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
model_name = "chatglm2-6b-32k"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载chatglm2-6b-32k模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model2path[model_name], trust_remote_code=True,
                                                  output_scores=True, return_dict_in_generate=True, 
                                                  torch_dtype=torch.bfloat16).to(device)

# 获取tokenizer的词表大小
vocab_size = model.config.padded_vocab_size

# 打印词表大小
print("The vocab size of chatglm2-6b-32k tokenizer is", vocab_size)

dataset = "konwledge_memorization"
max_gen = dataset2maxlen[dataset]

watermark_args = Namespace(mode="gpt", gamma=0.1, delta=10.0, initial_seed=1234, dynamic_seed=None, bl_type="soft", num_beams=1, sampling_temp=0.7)

generator = Generator(watermark_args, tokenizer, model)
with open("testfile.txt", "r", encoding="utf-8") as f:
    data = f.read()

# 对数据进行分词
input = tokenizer(data, truncation=False, return_tensors="pt").to(device)

completions_text, completions_tokens  = generator.generate(input_ids=input.input_ids, max_new_tokens=max_gen)

print("####################")
