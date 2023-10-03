import argparse
import json
import re
import os
import pandas as pd

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="avg")
    parser.add_argument('--threshold', type=int, default=4)
    return parser.parse_args(args)

def main(args):
    ### process_mutual
    df = pd.DataFrame(columns=["mode", "mission_name", "score"])

    input_dir = "./pred"

    model_name = "llama2-7b-chat-4k"
    num = 0
    p = r"(?P<model_name>.+)_(?P<mode>old|v2|gpt|new|no)_g(?P<gamma>.+)_d(?P<delta>\d+(\.\d+)?)"
    p2 = r"(?P<misson_name>[a-zA-Z_]+)_(?P<gamma>\d+(\.\d+)?)_(?P<delta>.+)_z"
    
    p1 = r"(?P<misson_name>[a-zA-Z_]+)_(?P<threshold>\d+(\.\d+)?)_z"

    # get all files from input_dir
    for subfolder in os.listdir(input_dir):
        print("subfolder is:", subfolder)
        if "human" in subfolder:
            continue
        matcher = re.match(p, subfolder)
        
        model_name = matcher.group("model_name")
        mode = matcher.group("mode")
        gamma = matcher.group("gamma")
        delta = matcher.group("delta")
        
        bl_type = "None"
        bl_type = (subfolder.split("_")[-1]).split(".")[0]
        
        if bl_type != "hard":
            if "old" in subfolder:
                bl_type = "soft"
            else:
                bl_type = "None"
            
        if bl_type == "hard" or bl_type == "soft":
            final_mode = model_name + "_" + mode + "_" + bl_type + "_" + "g"+gamma + "_" + "d" + delta  
        else:
            final_mode = model_name + "_" + mode + "_" + "g"+gamma + "_" + "d" + delta    
        
        # ref_mode = subfolder.split("ref_")[1]
            
        # print(ref_mode)
        eval_path = os.path.join(input_dir, subfolder, "eval")
        if os.path.exists(eval_path):
            result_file = os.path.join(eval_path, "result.json")
            if os.path.exists(result_file):
                with open (result_file, "r") as f:
                    data = json.load(f)
                keys = data.keys()
                temp_df = pd.DataFrame(columns=["mode", "mission_name", "score"])
                
                for key in keys:
                    temp_df = pd.DataFrame({
                        "mode":[final_mode],
                        "mission_name": [key],
                        "score":[(str(data[key])).split(" ")[-1]]
                    })
                    
                    df = pd.concat([df, temp_df], ignore_index=True)
                    
                    num += 1
    df = df.sort_values(by="mode", ascending=True)                 
    df.to_csv("eval.csv")                
    print("num is:", num)
                        
if __name__ == '__main__':
    args = parse_args()
    main(args)
     