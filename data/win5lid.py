import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Union


class Win5LID_datset(torch.utils.data.Dataset):
    """
    Loading and processing the Win5-LID dataset.
    """

    def __init__(self, folders: Union[List[str], str], transform: callable):
        super(Win5LID_datset, self).__init__()
        # Load the paths and quality scores
        self.dataset_dic = self.get_paths_score(folders)
        self.transform = transform

    def get_paths_score(self, folders: Union[List[str], str]):
        """
        Retrieves file paths and corresponding quality scores from the dataset.
        """
        root = "data/win5lidHor/"
        df = pd.read_excel("data/Win5-LID_MOS.xlsx")
        data_dic = {}
        idx = 0

        # If folders is a list, process each folder
        if isinstance(folders, list):
            for folder in folders:
                root_images = os.path.join(root, folder)
                for file in os.listdir(root_images):
                    file_name = file.split(".")[0]
                    value = df.loc[df["filename"] == file_name]["Picture_MOS"].values[0]
                    file_path = os.path.join(root_images, file)
                    data_dic[idx] = (file_path, value, file_name)
                    idx += 1

        # If folders is a single folder, process that folder
        else:
            root_images = os.path.join(root, folders)
            for file in os.listdir(root_images):
                file_name = file.split(".")[0]
                value = df.loc[df["filename"] == file_name]["Picture_MOS"].values[0]
                file_path = os.path.join(root_images, file)
                data_dic[idx] = (file_path, value, file_name)
                idx += 1

        return data_dic

    def __len__(self):
        return len(self.dataset_dic)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str]]:
        """
        Retrieves a sample from the dataset at the given index.
        """

        d_img_path, score, name = self.dataset_dic.get(idx, 0)
        image = Image.open(f"{d_img_path}")
        if self.transform:
            image = self.transform(image)

        sample = {"d_img_org": image, "score": score, "name": name}

        return sample
