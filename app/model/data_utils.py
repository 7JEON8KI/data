import pandas as pd
import numpy as np

""" 
    fileName      : data_utils.py
    author        : 이소민
""" 

def load_and_process_data():
    products = pd.read_csv('app/dataset/product_df_with_ingredients.csv')
    image_urls = []
    for url in products['THUMBNAIL_IMAGE_URL']:
        image_urls.append(url)
    return products, image_urls

def load_orders_data():
    df_order = pd.read_csv('app/dataset/df_orders.csv')
    return df_order