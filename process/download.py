from tqdm import tqdm
import numpy as np
import random
import argparse
import os
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import json
import hashlib

dataset2level = {
    "konwledge_memorization" : "1-1",
    "konwledge_understanding" : "1-2",
    "longform_qa" : "2-1",
    "finance_qa" : "2-2",
    "hotpotqa" : "3-1",
    "lcc" : "3-2",
    "multi_news" : "4-1",
    "qmsum" : "4-2",
    "alpacafarm" : "5-1",
}

origin_datasets = ["kola", "finance_qa", "eli5", "longbench", "alpacafarm"]

def convert_to_sha256(string):
    # Encoding the string into bytes
    encoded_string = string.encode('utf-8')

    # Creating a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Updating the hash object with the encoded string
    sha256_hash.update(encoded_string)

    # Obtaining the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="all", choices=["all"]+origin_datasets)
    return parser.parse_args(args)

def cal_len_and_output(_input, context, output, f, dataset):
    # calculate length
    input_total = _input + " " + context
    input_count = len(word_tokenize(input_total))
    if type(output) == list:
        output_count =  sum([ len(word_tokenize(o)) for o in output]) / len(output)
    else:
        output_count = len(word_tokenize(output))
    json_obj = {}
    json_obj["input"] = _input # The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc
    json_obj["context"] = context # The long context required for the task, such as documents, cross-file code, few-shot examples in Few-shot tasks
    if type(output) == list:
        json_obj["outputs"] = output
    else:
        json_obj["outputs"] = [output]
    json_obj["input_length"] = input_count
    json_obj["output_length"] = output_count
    json_obj["length"] = input_count + output_count
    # other keys
    json_obj["all_classes"] = "null"
    json_obj["language"] = "en"
    json_obj["dataset"] = dataset
    encode_str = "Tsinghua WaterBench " + json.dumps(json_obj)
    json_obj["_id"] = convert_to_sha256(encode_str)
    json.dump(json_obj, f)
    f.write("\n")
    

def get_data(dataset_name):
    output_dir = f"./data/WaterBench/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dataset_name == "longbench":
        datasets = [ "hotpotqa", "multi_news",  "lcc", "qmsum"]
        for dataset in datasets:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            print(data)
            output_path = f"{output_dir}/{dataset2level[dataset]}_{dataset}.jsonl"
            with open(output_path, "w") as f:
                if len(data) > 200:
                    data = data.shuffle(seed=42).select(range(200))
                for json_obj in data:
                    # calculate length
                    cal_len_and_output(json_obj["input"], json_obj["context"], json_obj["answers"][0], f, dataset)
    elif dataset_name == "alpacafarm":
        alapaca_eval_data = load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")["eval"]
        # load data and output
        output_path = f"{output_dir}/{dataset2level[dataset_name]}_{dataset_name}.jsonl"
        with open(output_path, "w") as fout:
            for json_obj in alapaca_eval_data:
                # calculate length
                cal_len_and_output(json_obj["input"], json_obj["instruction"], json_obj["output"], fout, dataset_name)
    elif dataset_name == "finance_qa":
        output_path = f"{output_dir}/{dataset2level[dataset_name]}_{dataset_name}.jsonl"
        # load
        with open(output_path, "w") as f:
            with open("/data2/tsq/WaterBench/data/download/HC3_en.jsonl", 'r') as fin:
                lines = fin.readlines()[19142:21142]
                json_with_len  = []
                for line in lines:
                    json_obj = json.loads(line.strip())
                    refs = json_obj["human_answers"]
                    output_count =  sum([ len(word_tokenize(o)) for o in refs]) / len(refs)
                    if output_count > 300 or "?" not in json_obj["question"]:
                        continue
                    json_with_len.append((json_obj, output_count))
                # sort revese
                json_with_len.sort(key=lambda x: x[1], reverse=True)
                for json_obj, _ in json_with_len[:200]:
                    question = json_obj["question"]
                    refs = json_obj["human_answers"]
                    cal_len_and_output(question, "", refs, f, dataset_name)   
    elif dataset_name == "eli5":
        dataset = "longform_qa"
        output_path = f"{output_dir}/{dataset2level[dataset]}_{dataset}.jsonl"
        with open(output_path, "w") as f:
            with open("/data2/tsq/WaterBench/data/download/HC3_en.jsonl", 'r') as fin:
                lines = fin.readlines()[:2000]
                json_with_len  = []
                for line in lines:
                    json_obj = json.loads(line.strip())
                    refs = json_obj["human_answers"]
                    output_count =  sum([ len(word_tokenize(o)) for o in refs]) / len(refs)
                    if output_count > 300:
                        continue
                    json_with_len.append((json_obj, output_count))
                # sort revese
                json_with_len.sort(key=lambda x: x[1], reverse=True)
                for json_obj, _ in json_with_len[:200]:
                    question = json_obj["question"]
                    refs = json_obj["human_answers"]
                    cal_len_and_output(question, "", refs, f, dataset)                    
    elif dataset_name == "kola":
        # need 1-1, 1-2. COPEN
        datasets = [ "konwledge_memorization", "konwledge_understanding"]
        for dataset in datasets:
            if dataset == "konwledge_memorization":
                data_paths = ["/data2/cookie/input/KG/1_low_freq_ent/test.json", "/data2/cookie/input/KG/2_high_freq_ent/test.json"]
            else:
                data_paths = ["/data2/cookie/input/IE/COPEN/cpj/test.json", "/data2/cookie/input/IE/COPEN/csj/test.json"]
            output_path = f"{output_dir}/{dataset2level[dataset]}_{dataset}.jsonl"
            with open(output_path, "w") as f:
                for data_path in data_paths:
                    data_file = json.load(open(data_path, 'r'))["request_states"]
                    for _instance in data_file:
                        instance = _instance["instance"]
                        input = instance["input"]["text"]
                        output = instance["references"][0]["output"]["text"]
                        context = ""
                        cal_len_and_output(input, context, output, f, dataset)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "all":
        for origin_dataset in origin_datasets:
            get_data(origin_dataset)
    else:
        get_data(args.dataset)