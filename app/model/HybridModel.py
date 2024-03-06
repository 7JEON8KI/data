import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
import urllib
import torchvision.transforms as transforms
import urllib.request
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_utils import load_and_process_data
products, image_urls = load_and_process_data()

with open("app/model/tfidf_vector.pkl", "rb") as f:
    tfidf_vector = pickle.load(f)

tfidf_matrix = tfidf_vector.fit_transform(products['food_ingredient']).toarray()
tfidf_matrix_feature = tfidf_vector.get_feature_names_out()
tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = products.product_name)

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index = products.product_name, columns = products.product_name)

def ingredient_recommendations_product_id(target_name, matrix, items, k=10):
    recom_idx = matrix.loc[:, target_name].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    recom_id = items.iloc[recom_idx, :].product_id.values
    return recom_id

IMG_HEIGHT = 512 
IMG_WIDTH = 512

def load_image_tensor(image_url, device):
    # 이미지 다운로드
    with urllib.request.urlopen(image_url) as response:
        with open('temp_image.jpg', 'wb') as out_file:
            out_file.write(response.read())

    # 이미지 열기 및 변환
    image = Image.open('temp_image.jpg').convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_tensor = transforms.ToTensor()(image)
    # 이미지 텐서를 배치 차원에 맞게 변환
    image_tensor = image_tensor.unsqueeze(0)
    # 임시 이미지 파일 삭제
    os.remove('temp_image.jpg')
    # 이미지 텐서를 GPU 또는 CPU 장치로 전송
    image_tensor = image_tensor.to(device)
    return image_tensor

def compute_similar_images(image_path, num_images, embedding, device):
    image_tensor = load_image_tensor(image_path, device)
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    print(indices_list)
    return indices_list

def get_product_ids(indices_list, products):
    similar_product_ids = []

    for sublist in indices_list:
        if not sublist:
            print(f"Skipping empty sublist.")
            continue

        for index in sublist:
            try:
                product_id = products.iloc[index]['product_id']
                similar_product_ids.append(product_id)

            except IndexError:
                print(f"Invalid index: {index}. Skipping...")

    return similar_product_ids

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
encoder = ConvEncoder()

# Load the state dict of encoder
encoder.load_state_dict(torch.load("app/model/baseline_encoder.pt", map_location=device))
encoder.eval()
encoder.to(device)

# Loads the embedding
embedding = np.load("app/model/data_embedding_f.npy")

def hybrid_recommender(target_name, target_image_path, products=products, content_matrix=cosine_sim_df, embedding=embedding, device=device, k=5):
    """하이브리드 추천 시스템 함수"""

    # 콘텐츠 기반 추천 결과
    content_recommendations = ingredient_recommendations_product_id(target_name, content_matrix, products, k)

    # 이미지 기반 추천 결과
    image_indices = compute_similar_images(target_image_path, k, embedding, device)
    image_recommendations = get_product_ids(image_indices, products)[:k]  # 최상위 k개만 선택

    # 콘텐츠 기반 추천과 이미지 기반 추천의 결과를 결합
    combined_recommendations = pd.DataFrame({'recom_id': content_recommendations, 'recommendation_score': 1})

    # 이미지 기반 추천 결과를 반영하여 추천 점수 업데이트
    for recom_id in image_recommendations:
        if recom_id in combined_recommendations['recom_id'].values:
            combined_recommendations.loc[combined_recommendations['recom_id'] == recom_id, 'recommendation_score'] += 1
        else:
            combined_recommendations = pd.concat([combined_recommendations, pd.DataFrame({'recom_id': [recom_id], 'recommendation_score': [1]})], ignore_index=True)

    # 추천 점수에 따라 정렬
    combined_recommendations = combined_recommendations.sort_values(by='recommendation_score', ascending=False)

    # recom_id에 해당하는 product 데이터를 데이터프레임 형태로 반환
    recommended_products = pd.DataFrame()
    for recom_id in combined_recommendations['recom_id'][:k]:  # 최상위 k개만 선택
        recommended_products = pd.concat([recommended_products, products[products['product_id'] == recom_id]], ignore_index=True)

    # 데이터프레임을 JSON 형태로 변환
    result = []
    for index, row in recommended_products.iterrows():
        result.append({
            "productId": str(row['product_id']),
            "productName": str(row['product_name']),
            "price": int(row['price']),
            "mainImgUrl": str(row['image_main']),
            "productType" : str(row['productType']), 
            "discountRate" : int(row['discountRate'])
        })    
    return result


# TARGET_PRODUCT = '청정원 호밍스 밀키트 부산식 곱창전골 760g x 2개'
# TARGET_IMAGE_PATH = "https://image.hmall.com/static/9/9/26/27/2127269978_0.jpg?RS=520x520&AR=0&ao=2"
# NUM_RECOMMENDATIONS = 10
# WEIGHT_CONTENT = 0.5
# WEIGHT_IMAGE = 0.5

# # 예시 사용
# hybrid_recommendations = hybrid_recommender(TARGET_PRODUCT, TARGET_IMAGE_PATH, products, cosine_sim_df, embedding, device, NUM_RECOMMENDATIONS)
# print(hybrid_recommendations)
