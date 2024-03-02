# data_utils.py
import pandas as pd
import numpy as np

def load_and_process_data():
    product = pd.read_csv('app/dataset/preprocessing_product.csv')
    product.rename(columns={'product_price': 'price'}, inplace=True)
    product['productType'] = 'mealkit'
    product['discountRate'] = np.random.choice([5, 10, 20, 30], size=len(product))

    products = pd.read_csv('app/dataset/hmall_food_last.csv')
    image_df = products[['image_main']]

    products = pd.concat([product, image_df], axis=1)
    image_urls = []

    for url in products['image_main']:
        image_urls.append(url)

    return products, image_urls
