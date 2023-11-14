import argparse
import json
import re
import os
import pandas as pd
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="internlm-7b-8k")
    parser.add_argument('--task', type=str, default="avg")
    parser.add_argument('--threshold', type=int, default=4)
    return parser.parse_args(args)


def main(args):

    df = pd.DataFrame(columns=["model_name", "mode", "gamma", "delta", "threshold", "bl_type", "z_score", "true_positive", "false_negative","sum"])

    input_dir = "./pred"
    p = r"(?P<model_name>.+)_(?P<mode>old|v2|gpt|new|no)_g(?P<gamma>.+)_d(?P<delta>\d+(\.\d+)?)"
    p1 = r"(?P<misson_name>[a-zA-Z_]+)_(?P<gamma>\d+(\.\d+)?)_(?P<delta>.+)_z"

    num = 0
    # get all files from input_dir
    for subfolder in os.listdir(input_dir):
        # print("subfolder is:", subfolder)
        matcher = re.match(p, subfolder)
        if matcher == None:
            continue
        model_name = matcher.group("model_name")
        # if model_name != "tulu-7b":
        if model_name != args.model:
        # if model_name != "tulu-7b":
            continue
        mode = matcher.group("mode")
        gamma = matcher.group("gamma")
        delta = matcher.group("delta")
        
        bl_type = "None"
        bl_type = (subfolder.split("_")[-1]).split(".")[0]
        
        print("bl_type", bl_type)
        if bl_type != "hard":
            if "old" in subfolder:
                bl_type = "soft"
            else:
                bl_type = "None"
            
            
        # print(model_name, mode, gamma, delta, bl_type)
        
        z_score_path = os.path.join(input_dir, subfolder, "z_score")
        if os.path.exists(z_score_path):
            
            print("subfolder is:", subfolder)
            for threshold in np.arange(0, 10, 0.1):
                files = os.listdir(z_score_path)
                temp_df = pd.DataFrame(columns=["model_name", "mode", "gamma", "delta", "threshold", "bl_type", "z_score", "sum"])
                all_z = []
                sums = []
                tp = 0
                fn = 0
                for file in files:
                    # print(file)
                    # read jsons
                    # matcher1 = re.match(p1, file)
                    # if matcher1:
                    #     misson_name = matcher1.group("misson_name")
                        # threshold = 4.0
                    

                    with open(os.path.join(z_score_path, file), "r") as f:
                        data = json.load(f)
                    # calculate tp and fn
                    # threshold = args.threshold
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
                true_positive = tp / sum(sums)
                if bl_type == "hard":
                    print("threshold:", threshold)
                    print("true_positive", true_positive)
                if true_positive >= 0.69 and true_positive <= 0.71:
                    # if bl_type == "hard":
                    #     print("hello1")
                    temp_df = pd.DataFrame({
                            "model_name": [model_name],
                            "mode": [mode],
                        
                            "gamma": [gamma],
                            "delta":[delta],
                            "threshold": [threshold],
                            "bl_type": [bl_type],
                            "z_score": [sum(all_z)/sum(sums)],
                            "true_positive": [tp/sum(sums)], 
                            "false_negative": [fn/sum(sums)],
                            "sum": [sum(sums)]})
                    df = pd.concat([df, temp_df], ignore_index=True)
                    
    # df = df.sort_values(by="threshold", ascending=True)     
            # df.drop(columns=["mission_name"], inplace=True)   
    df.to_csv(f"csv_data/mod_threshold_{args.model}2.csv") 
    
    deltas = [2, 5, 10, 15]
    gammas = []
    if 'llama' in args.model: 
        gammas = [0.1, 0.25, 0.5, 0.75, 0.9]
    elif 'internlm' in args.model:
        gammas = [0.1, 0.15, 0.25, 0.5, 0.75, 0.9]    
    
    for gamma in gammas:
        # true_po = df[(df['bl_type'] == 'hard') & (df['gamma'] == str(gamma))]['true_positive']
        sub_df = df[(df['bl_type'] == 'hard') & (df['gamma'] == str(gamma))]
        
        # print("true_po is", true_po)
        print("len sub_df is", len(sub_df['true_positive']))
        if len(sub_df['true_positive']) != 0:
            # print("1 point")
            for delta in deltas:
                # print(sub_df['delta'].values.astype(float))
                if (float(delta) not in sub_df['delta'].values.astype(float)):
                    print("2 point")
                    temp_df = pd.DataFrame({
                    "model_name": [sub_df['model_name'].values[0]],
                    "mode": ['old'],
                    "gamma": [gamma],
                    "delta":[delta],
                    "threshold": [threshold],
                    "bl_type": ['hard'],
                    "z_score": [sub_df['z_score'].values[0]],
                    "true_positive": [sub_df['true_positive'].values[0]], 
                    "false_negative": [sub_df['false_negative'].values[0]],
                    "sum": [sub_df['sum'].values[0]]})

                    df = pd.concat([df, temp_df], ignore_index=True)
    
    df = df.sort_values(by="threshold", ascending=True)    
    df.to_csv(f"csv_data/mod_threshold_{args.model}.csv") 
            
    print(df)
    print(num)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
     