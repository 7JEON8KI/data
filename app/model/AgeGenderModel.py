from .data_utils import load_and_process_data
from .data_utils import load_orders_data

""" 
    fileName      : AgeGenderModel.py
    author        : 이소민
""" 

df_products, _ = load_and_process_data()
df_orders = load_orders_data()

popular_products_by_group = df_orders.groupby(['age_group', 'member_gender'])['product_id'].value_counts().groupby(level=[0, 1]).head(5)

def recommend_by_age_gender(age_group, gender):
    gender_str = "남성" if gender == 1 else "여성"
    recommendations = popular_products_by_group.loc[(age_group, gender)]
    result = [
        {
            "ment" : str(age_group) + "대 " + gender_str + "에게 인기 많은 밀킷",
            "products": []
        }
    ]
    for product_id in recommendations.index:
        product = df_products[df_products['PRODUCT_ID'] == product_id].iloc[0]
        result[0]["products"].append({
            "productId": str(product['PRODUCT_ID']),
            "productName": str(product['PRODUCT_NAME']),
            "price": int(product['PRICE']),
            "mainImgUrl": str(product['THUMBNAIL_IMAGE_URL']),
            "productType" : str(product['PRODUCT_TYPE']), 
            "discountRate" : int(product['DISCOUNT_RATE'])
        })
    return result