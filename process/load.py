from process.download import dataset2level
from typing import List, Dict
import pandas as pd
import json

class DataLoader():
    def __init__(self, dataset_names: list) -> None:
        if dataset_names == "all":
            dataset_names = list(dataset2level.keys())
        self.dataset_names = dataset_names
    
    def load_data(self) -> List[Dict]:
        data = []
        for dataset_name in self.dataset_names:
            data_path = "./data/WaterBench/" + dataset2level[dataset_name] + "_" + dataset_name + ".jsonl"
            # load
            with open(data_path, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
        return data

if __name__ == '__main__':
    dataloader = DataLoader("all")
    for name in dataloader.dataset_names:
        print(name)
        new_loader = DataLoader([name])
        datasets = new_loader.load_data()
        df = pd.DataFrame(datasets)
        # aggeregate the "input_length", "output_length", and "length"
        print(df.describe())

    