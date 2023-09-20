import json
import os
import pandas as pd

input_dir = './data/WaterBench'
df = pd.DataFrame(columns=["prompt", "pred", "completions_tokens", "answers", "all_classes", "length"])
save_dir = './pred/human_generation'
os.makedirs("./pred/human_generation", exist_ok=True)


for subfile in os.listdir(input_dir):
    with open(os.path.join(input_dir, subfile), 'r') as f:
        df = pd.DataFrame(columns=["prompt", "pred", "completions_tokens", "answers", "all_classes", "length"])
        for line in f:
            data = json.loads(line)
            outputs = data['outputs']
            input = data['context']
            context = data['context']
            length = data['length']
            all_classes = data['all_classes']
            
            temp = pd.DataFrame({
                "prompt": [input],
                "pred": [outputs[0]],
                "completions_tokens": [""],
                "answers": [outputs],
                "all_classes": [all_classes],
                "length": [length] 
            })
            
            df = pd.concat([df, temp], ignore_index=True)
        result_file = "_".join(subfile.split("_")[1:])
        df_dict = df.to_dict(orient='records')
        
        out_path = os.path.join(save_dir, result_file)
        with open(out_path, 'w') as f:
            for row in df_dict:
                f.write(json.dumps(row) + '\n')
            
        
        
        
            
            
