import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import torch.nn as nn

import torchvision.transforms as transforms

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data_utils import load_and_process_data

""" 
    fileName      : ImageSearch.py
    author        : 이소민
""" 

products, image_urls = load_and_process_data()

NUM_IMAGES = 10
ENCODER_MODEL_PATH = "app/model/baseline_encoder.pt"
DECODER_MODEL_PATH = "app/model/baseline_encoder.pt"
EMBEDDING_PATH = "app/model/data_embedding_f.npy"
EMBEDDING_SHAPE = (1, 256, 16, 16)

class FolderDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image
    
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.img_size = img_size
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        return x
    
class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)
        return x

embedding = np.load(EMBEDDING_PATH)

def compute_similar_images(image_tensor, num_images=NUM_IMAGES, embedding=embedding, device="cpu"):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    print(indices_list)
    return indices_list

def get_products(indices_list, products):
    similar_products = []
    for sublist in indices_list:
        if not sublist:
            print(f"Skipping empty sublist.")
            continue
        for index in sublist:
            try:
                product = products.iloc[index-1]
                similar_products.append(product)
            except IndexError:
                print(f"Invalid index: {index}. Skipping...")
    return similar_products

encoder = ConvEncoder()

def image_processing_and_search(image_file, IMG_HEIGHT=512, IMG_WIDTH=512, device="cpu"):
    img_resized = image_file.resize((IMG_WIDTH, IMG_HEIGHT))

    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    img_tensor = img_tensor[:, :3, :, :]

    encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Loads the embedding
    embedding = np.load(EMBEDDING_PATH)

    indices_list = compute_similar_images(img_tensor, NUM_IMAGES, embedding, device)
    similar_products = get_products(indices_list, products)
    
    print("indices_list : ", indices_list)
    similar_product_ids = [product['PRODUCT_ID'] for product in similar_products]
    print(similar_product_ids)

    result = []
    for product in similar_products:
        result.append({
            "productId": str(product['PRODUCT_ID']),
            "productName": str(product['PRODUCT_NAME']),
            "price": str(product['PRICE']),
            "mainImgUrl": str(product['THUMBNAIL_IMAGE_URL']),
            "productType" : str(product['PRODUCT_TYPE']), 
            "discountRate" : int(product['DISCOUNT_RATE'])
        })    
    return result
