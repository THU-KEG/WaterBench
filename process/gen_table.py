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
    # category: L1, L2, L3, L4, L5
    df = pd.DataFrame(columns=["mode", "category", "true_positive", "true_negative", "result", "drop"])
    
    df_eval = pd.read_csv(io="./eval.csv")
    
    mode0 = "llama2-7b-chat-4k_no_g0.5_d5.0"
    mode1 = "llama2-7b-chat-4k_v2_g0.25_d15.0"
    mode2 = "llama2-7b-chat-4k_old_soft_g0.1_d10.0"
    mode3 = "llama2-7b-chat-4k_old_hard_g0.25_d5.0"
    mode4 = "llama2-7b-chat-4k_gpt_g0.1_d10.0"

    input_dir = "./pred"
    p = r"(?P<model_name>.+)_(?P<mode>old|v2|gpt|new|no)_g(?P<gamma>.+)_d(?P<delta>\d+(\.\d+)?)"
    p1 = r"(?P<misson_name>[a-zA-Z_]+)_(?P<gamma>\d+(\.\d+)?)_(?P<delta>.+)_z"

    num = 0
    # get all files from input_dir
    for subfolder in os.listdir(input_dir):
        # print("subfolder is:", subfolder)
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
            
        # print(model_name, mode, gamma, delta, bl_type)
        
        z_score_path = os.path.join(input_dir, subfolder, "z_score")
        if os.path.exists(z_score_path):
            print("subfolder is:", subfolder)
            files = os.listdir(z_score_path)
            temp_df = pd.DataFrame(columns=["model_name", "mission_name", "mode", "gamma", "delta", "threshold", "bl_type", "z_score", "sum"])
            all_z = []
            sums = []
            tp = 0
            fn = 0
            for file in files:
                # print(file)
                # read jsons
                matcher1 = re.match(p1, file)
                if matcher1:
                    misson_name = matcher1.group("misson_name")
                    threshold = 6.0
                else:
                    threshold = file.split("_")[-2]
                
                with open(os.path.join(z_score_path, file), "r") as f:
                    data = json.load(f)
                # calculate tp and fn
                threshold = args.threshold
                z_score_list = data["z_score_list"]
                _sum = len(data["z_score_list"])
                tp += len([x for x in z_score_list if x > threshold])
                # print("tp is:", tp)
                fn += len([x for x in z_score_list if x <= threshold])
                # print("fn is:", fn)
                
                # get data
                avarage_z = data["avarage_z"]
                all_z.append(avarage_z * _sum)
                sums.append(_sum)
                num += 1

            # average z_score
            # print(temp_df)
            temp_df = pd.DataFrame({
                    "model_name": [model_name],
                    "mode": [mode],
                    "mission_name": [misson_name],
                    "gamma": [gamma],
                    "delta":[delta],
                    "threshold": [threshold],
                    "bl_type": [bl_type],
                    "z_score": [sum(all_z)/sum(sums)],
                    "true_positive": [tp/sum(sums)], 
                    "false_negative": [fn/sum(sums)],
                    "sum": [sum(sums)]})
            df = pd.concat([df, temp_df], ignore_index=True)
    df = df.sort_values(by="true_positive", ascending=True)     
    df.drop(columns=["mission_name"], inplace=True)   
    df.to_csv("z_score_avg.csv") 
            
    print(df)
    print(num)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
     