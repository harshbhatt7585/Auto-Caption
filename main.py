from BLIP.models.blip import blip_decoder
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 384

def load_model():
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    return model

def predict(image):
    with torch.no_grad():
        # beam search
        # caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        des_path = os.path.join('/workspace/imgs', img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            os.remove(os.path.join(self.root_dir, img_name))
            print('removed', img_name)

        if self.transform:
            image = self.transform(image)
            

        return image, img_name


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--data_path', required=True)
    args.add_argument('--csv_path', required=True, default='metadata.csv')

    args = args.parse_args()

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    
    dataset = CustomImageDataset(root_dir='/workspace/part3', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = load_model()

    for idx, (img, img_name) in enumerate(dataloader):
        with torch.no_grad():
            caption = model.generate(img.to(device), sample=True, top_p=0.9, max_length=20, min_length=5)
            # Writing data to the CSV file
            with open(args.csv_path, mode='a', newline='') as csvfile:
                fieldnames = ["file_name", "text"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if(idx ==0):   
                    # Write the header row
                    writer.writeheader()
                for i in range(len(caption)):
                    row = {'file_name': img_name[i], 'text': caption[i] }
                    writer.writerow(row)
