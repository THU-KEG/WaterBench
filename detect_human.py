from watermark.old_watermark import OldWatermarkDetector
from watermark.our_watermark import NewWatermarkDetector
from watermark.gptwm import GPTWatermarkDetector
from watermark.watermark_v2 import WatermarkDetector
from tqdm import tqdm
from pred import load_model_and_tokenizer, seed_everything, str2bool
import argparse
import os
import json
import torch
import re

def main(args):
    seed_everything(42)
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model name
    model_name = args.reference_dir.split("/")[-1].split("_")[0]
    
    # define your model
    tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, load_token_only=True)
    all_token_ids = list(tokenizer.get_vocab().values())
    vocab_size = len(all_token_ids)
    
    
    all_input_dir = "./pred/"
    # get gamma and delta
    
    pattern_dir = r"(?P<model_name>.+)_(?P<mode>old|v2|gpt|new|no)_g(?P<gamma>.+)_d(?P<delta>\d+(\.\d+)?)"
    
    pattern_mis = r"(?P<misson_name>[a-zA-Z_]+)_(?P<gamma>\d+(\.\d+)?)_(?P<delta>.+)_z"
    
    matcher_ref = re.match(pattern_dir, args.reference_dir)
    
    mode_ref = matcher_ref.group("mode")
    gamma_ref = float(matcher_ref.group("gamma"))
    delta_ref = float(matcher_ref.group("delta"))
    bl_type_ref = "None"
    bl_type_ref = (args.reference_dir.split("_")[-1]).split(".")[0]
    
    if bl_type_ref != "hard":
        if "old" in args.reference_dir:
            bl_type_ref = "soft"
            mode_ref += "_" + bl_type_ref
        else:
            bl_type_ref = "None"   
    else:
        mode_ref += "_" + bl_type_ref
    
    print("mode_ref is:", mode_ref)  
    
    if args.detect_dir != "human_generation":
        matcher_det = re.match(pattern_dir, args.detect_dir)
        mode_det = matcher_det.group("mode")
        gamma_det = float(matcher_det.group("gamma"))
        delta_det = float(matcher_det.group("delta"))
        bl_type_det = "None"
        bl_type_det = (args.detect_dir.split("_")[-1]).split(".")[0]


        if bl_type_det != "hard":
            if "old" in args.detect_dir:
                bl_type_det = "soft"
                mode_det += "_" + bl_type_det
            else:
                bl_type_det = "None"
        else:
            mode_det += "_" + bl_type_det

        # print("bl_type_det is:", bl_type_det)  
        # print("mode_det is:", mode_det)    
    # get all files from detect_dir
    
    files = os.listdir(all_input_dir + args.detect_dir)
    
    # get all json files
    json_files = [f for f in files if f.endswith(".jsonl")]
    
    ref_dir = f"./detect_human/{model_name}/ref_{mode_ref}_g{gamma_ref}_d{delta_ref}"
    
    os.makedirs(f"./detect_human/{model_name}", exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    if args.detect_dir == "human_generation":
          os.makedirs(ref_dir + "/human_generation_z", exist_ok=True)
    else:
        os.makedirs(ref_dir + f"/{mode_det}_g{gamma_det}_d{delta_det}_z", exist_ok=True)
    
    if "old" in args.reference_dir or "no" in args.reference_dir:
        detector = OldWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma_ref,
                                            delta=delta_ref,
                                            dynamic_seed="markov_1",
                                            device=device)
        
    if "new" in args.reference_dir:
        detector = NewWatermarkDetector(tokenizer=tokenizer,
                                    vocab=all_token_ids,
                                    gamma=gamma_ref,
                                    delta=delta_ref,
                                    dynamic_seed="markov_1",
                                    device=device,
                                    # vocabularys=vocabularys,
                                    )
        
    if "v2" in args.reference_dir:
        detector = WatermarkDetector(
            vocab=all_token_ids,
            gamma=gamma_ref,
            z_threshold=args.threshold,tokenizer=tokenizer,
            seeding_scheme=args.seeding_scheme,
            device=device,
            normalizers=args.normalizers,
            ignore_repeated_bigrams=args.ignore_repeated_bigrams,
            select_green_tokens=args.select_green_tokens)
        
    if "gpt" in args.reference_dir:
        detector = GPTWatermarkDetector(
            fraction=gamma_ref,
            strength=delta_ref,
            vocab_size=vocab_size,
            watermark_key=args.wm_key)
    prompts = []        
    for json_file in json_files:
        print(f"{json_file} has began.........")
        if args.detect_dir == "human_generation":
            with open(os.path.join(all_input_dir + args.reference_dir, json_file), "r") as f:
                lines = f.readlines()
                
                prompts = [json.loads(line)["prompt"] for line in lines]
                print("len of prompts is", len(prompts))
        # read jsons
        with open(os.path.join(all_input_dir + args.detect_dir, json_file), "r") as f:
            # lines
            lines = f.readlines()
            # texts
            if args.detect_dir != "human_generation":
                prompts = [json.loads(line)["prompt"] for line in lines]
            texts = [json.loads(line)["pred"] for line in lines]
            print(f"texts[0] is: {texts[0]}")
            tokens = [json.loads(line)["completions_tokens"] for line in lines]
                
        z_score_list = []
        for idx, cur_text in tqdm(enumerate(texts), total=len(texts)):
            #print("cur_text is:", cur_text)
            
            gen_tokens = tokenizer.encode(cur_text, return_tensors="pt", truncation=True, add_special_tokens=False)
            #print("gen_tokens is:", gen_tokens)
            prompt = prompts[idx]
            
            input_prompt = tokenizer.encode(prompt, return_tensors="pt", truncation=True,add_special_tokens=False)
            
            
            if len(gen_tokens[0]) >= args.test_min_tokens:
                
                if "v2" in args.reference_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                        
            if len(gen_tokens[0]) >= 1:
                if "gpt" in args.reference_dir:
                    z_score_list.append(detector.detect(gen_tokens[0]))
                    
                elif "old" in args.reference_dir or "no" in args.reference_dir:
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                    
                elif "new" in args.reference_dir:
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
            
            else:        
                print(f"Warning: sequence {idx} is too short to test.")
            
        save_dict = {
            'z_score_list': z_score_list,
            'avarage_z': torch.mean(torch.tensor(z_score_list)).item(),
            'wm_pred': [1 if z > args.threshold else 0 for z in z_score_list]
            }
        
        wm_pred_average = torch.mean(torch.tensor(save_dict['wm_pred'], dtype=torch.float))
        save_dict.update({'wm_pred_average': wm_pred_average.item()})   
        
        print(save_dict)
        # average_z = torch.mean(z_score_list)
        z_file = json_file.replace('.jsonl', f'_{args.threshold}_z.jsonl')
        
        if args.detect_dir != "human_generation":
            output_path = os.path.join(ref_dir + f"/{mode_det}_g{gamma_det}_d{delta_det}_z", z_file)
        
        else:
            output_path = os.path.join(ref_dir + "/human_generation_z", z_file)
        with open(output_path, 'w') as fout:
            json.dump(save_dict, fout)
            
            
        


parser = argparse.ArgumentParser(description="Process watermark to calculate z-score for every method")

parser.add_argument(
    "--input_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_old_g0.5_d5.0")

parser.add_argument( # for gpt watermark
        "--wm_key", 
        type=int, 
        default=0)

parser.add_argument(
    "--threshold",
    type=float,
    default=4.0)

parser.add_argument(
    "--test_min_tokens",
    type=int, 
    default=2)

parser.add_argument( # for v2 watermark
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )

parser.add_argument( # for v2 watermark
    "--normalizers",
    type=str,
    default="",
    help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
)

parser.add_argument( # for v2 watermark
    "--ignore_repeated_bigrams",
    type=str2bool,
    default=False,
    help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
)

parser.add_argument( # for v2 watermark
    "--select_green_tokens",
    type=str2bool,
    default=True,
    help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
)

parser.add_argument(
    "--reference_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_v2_g0.25_d15.0",
    help="Which type as reference to test TN or FP",
)

parser.add_argument(
    "--detect_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_v2_g0.25_d15.0",
    help="Which type need to be detected",
)
args = parser.parse_args()

main(args)

