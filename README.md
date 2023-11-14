# WaterBench
Data and Code for the paper, WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models  

## Installation
To download the repo to the local machine, run the following command:
``` bash
git clone https://github.com/THU-KEG/WaterBench.git
cd WaterBench
```

## How to evaluate on WaterBench

#### Load Data

Our datasets are stored in the `data` folder. To load the data, you can use the following code(Here we take the *alpacafarm* dataset as an example):

``` python
dataset2level = json.load(open("config/dataset2level.json", "r"))
dataset = "alpacafarm"
data = []
with open("data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
```

To convert the original input to the prompt format, you can use the following code(Here we take the *alpacafarm* dataset as an example):

``` python
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset = "alpacafarm"
prompt_format = dataset2prompt[dataset]

prompt = prompt_format.format(**json_obj)
# json_obj is every piece of the data
```

#### Data Format

All the missions WaterBench used are specified in the `data` folder. The data is in the format of `jsonl`. Each line is a json object with the following fields:

``` json
{
    "input": "The input/command for the task, usually short, such as questions in QA etc",
    "context": "The long context required for the task, such as documents, cross-file code",
    "outputs": "A List of all true answers",
    "input_length": "length of the first two items(counted in characters for Chinese and words for English)",
    "output_length": "length of the third item (counted in characters for Chinese and words for English)",
    "length": "length of the first three items(counted in characters for Chinese and words for English)",
    "all_classes": "All categories in classification tasks, null for non-classification tasks",
    "language": "The language of this piece of data",
    "dataset": "The name of the dataset to which this piece of data belongs",
    "_id": "Random id for each piece of data"
}
```

#### Load Model

The models we used are *llama2-7b-chat-4k* and *internlm-chat-7b-8k*, you can change the path of model in `config/model2path.json` file.

You can get the *llama2* from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), and *internlm* from [here](https://huggingface.co/internlm/internlm-chat-7b-8k).

We use the `transformers` library to load the model. The explicit code is in the `pred.py` file, and the following is the main code:

``` python
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
def load_model_and_tokenizer(path_to_your_model, model_name, device,  load_token_only=False):
    if "internlm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path_to_your_model, trust_remote_code=True)
        if not load_token_only:
            model = AutoModelForCausalLM.from_pretrained(path_to_your_model, trust_remote_code=True,
                                                  output_scores=True, return_dict_in_generate=True, 
                                                  torch_dtype=torch.bfloat16).to(device)
            model.eval()
    elif "llama2"in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path_to_your_model)
        if not load_token_only:
            model = LlamaForCausalLM.from_pretrained(path_to_your_model, output_scores=True, return_dict_in_generate=True, 
                                                 torch_dtype=torch.bfloat16).to(device) 
    if load_token_only:
        return tokenizer
    else:
        model = model.eval()
        return model, tokenizer

```

#### Evaluation

Install the requirements with pip: `pip install -r requirements.txt`. For Llama-2 based models, we recommend using Flash Attention for optimization and saving GPU memory The relevant dependencies can be installed according to the code base of [Flash Attention](https://github.com/Dao-AILab/flash-attention).

First, run `pred.py` and select the model you want to evaluate via `--model`. And select the mode and hyper-parameters of the watermark via `--mode`, `--bl_type`, `--gamma`, `--delta`. The parameter `mode` means the kinds of watermarks we used in the experiments, including `no`(without watermark), `old`(soft or hard watermark), `gpt`(gpt watermark), `v2`(v2, watermark). The parameter `bl_type` means whether the type of the watermark is hard or soft. Also, you can select the dataset you want to evaluate via `--dataset`. Here is an example to obtain the result that *llama2-7b-chat-4k* model performs with specified-parameters watermark on the *alpacafarm* dataset:

``` bash
CUDA_VISIBLE_DEVICES=0 python pred.py \
    --mode old \
    --gamma 0.5 \
    --delta 10 \
    --bl_type hard \ 
    --dataset alpacafarm \
    --model llama2-7b-chat-4k \
```

If you didn't specify the `--dataset`, the code will evaluate the model on all datasets in WaterBench.

You can obtain the output of the model under all WaterBench datasets under the `pred/` folder corresponding to the model name.

After that, run the detection code in `detect.py` to obtain z-scores:

``` bash
CUDA_VISIBLE_DEVICES=0 python detect.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.5_d10.0_hard
```

Then, you can obtain z-scores of every mission under the input_dir of detect .

After that, you can run the code in `eval.py` to obtain the gpt-4 evaluation results on all datasets in `result.json`:

``` bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --input_dir ./pred/llama2-7b-chat-4k_old_g0.5_d10.0_hard
```

Please note that in `config/`, we provide the input format suitable for each dataset and the maximum output length. Feel free to modify them to better suit the model you want to evaluate. After modification, when evaluating with `pred.py`, the data will be automatically organized according to the new format to get the corresponding model output.

