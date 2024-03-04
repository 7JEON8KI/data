from .data_utils import load_and_process_data
from .data_utils import load_orders_data

df_products, _ = load_and_process_data()
df_orders = load_orders_data()

popular_products_by_group = df_orders.groupby(['age_group', 'member_gender'])['product_id'].value_counts().groupby(level=[0, 1]).head(5)

def recommend_by_age_gender(age_group, gender):
    recommendations = popular_products_by_group.loc[(age_group, gender)]
    result = {
        "ageGroup": age_group,
        "gender": gender,
        "products": []
    }
    for product_id in recommendations.index:
        product = df_products[df_products['product_id'] == product_id].iloc[0]
        result["products"].append({
            "productId": str(product['product_id']),
            "productName": str(product['product_name']),
            "price": str(product['price']),
            "mainImgUrl": str(product['image_main']),
            "productType" : str(product['productType']), 
            "discountRate" : int(product['discountRate'])
        })
    return result