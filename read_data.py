from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path=os.listdir(self.path)

    def __getitem__(self, idx):
        image_name=self.image_path[idx]
        image=Image.open(os.path.join(self.path,image_name))
        lable=self.label_dir
        return image, lable

    def __len__(self):
        return len(self.image_path)