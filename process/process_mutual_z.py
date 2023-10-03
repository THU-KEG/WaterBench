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
    df = pd.DataFrame(columns=["model_name", "mission_name", "ref_mode", "det_mode", "threshold", "z_score", "sum"])

    input_dir = "./mutual_detect/llama2-7b-chat-4k"

    model_name = "llama2-7b-chat-4k"
    num = 0
    p1 = r"(?P<misson_name>[a-zA-Z_]+)_(?P<threshold>\d+(\.\d+)?)_z"

    # get all files from input_dir
    for subfolder in os.listdir(input_dir):
        print("subfolder is:", subfolder)
        
        ref_mode = subfolder.split("ref_")[1]
            
        # print(ref_mode)
        
        for subsubfolder in os.listdir(os.path.join(input_dir,subfolder)):
            
            det_mode = subsubfolder.split("_z")[0]
            print(det_mode)
        
            z_score_path = os.path.join(input_dir, subfolder, subsubfolder)
            if os.path.exists(z_score_path):
            # print("subfolder is:", subfolder)
                files = os.listdir(z_score_path)
                tn = 0
                fp = 0
                all_z = []
                sums = []
                for file in files:
                    # print(file)
                    # read jsons
                    matcher1 = re.match(p1, file)
                    if matcher1:
                        misson_name = matcher1.group("misson_name")
                        threshold = matcher1.group("threshold")
                    with open(os.path.join(z_score_path, file), "r") as f:
                        data = json.load(f)

                    
                    avarage_z = data["avarage_z"]
                    # sum = len(data["z_score_list"])
                    # threshold = args.threshold
                    z_score_list = data["z_score_list"]
                    # wm_pred_avarage = data["wm_pred_avarage"]
                    # calculate np and tn
                    threshold = args.threshold
                    z_score_list = data["z_score_list"]
                    _sum = len(data["z_score_list"])
                    tn += len([x for x in z_score_list if x > threshold])
                    
                    # print("tp is:", tp)
                    fp += len([x for x in z_score_list if x <= threshold])
                    
                    all_z.append(avarage_z * _sum)
                    sums.append(_sum)
                    num += 1
                    
                    
                    
                temp = pd.DataFrame({
                        "model_name": [model_name],
                        "ref_mode": [ref_mode],
                        "mission_name": [misson_name],
                        "det_mode":[det_mode],
                        "threshold": [threshold],
                        "z_score": [sum(all_z)/sum(sums)],
                        "false_positive": [fp/sum(sums)], 
                        "true_negative": [tn/sum(sums)],
                        "sum": [sum(sums)]})
                    
                df = pd.concat([df, temp], ignore_index=True)
                    
    df = df.sort_values(by="false_positive", ascending=True) 
    df.drop(columns=["mission_name"], inplace=True)
    df.to_csv("mutual_z_score_avg.csv")           
    print(df)
    print(num)

if __name__ == '__main__':
    args = parse_args()
    main(args)
     
