from fastapi import APIRouter
from pydantic import BaseModel

from .model.HybridModel import hybrid_recommender

router = APIRouter(prefix="/ai", tags=["ai"])

class TxtItem(BaseModel):
    content: str

class ProductItem(BaseModel):
    productName: str
    productMainImage: str

@router.post("/hybrid-recommendations")
async def calculate_recommendations_scores(item: ProductItem):
    target_product = item.productName
    target_image_path = item.productMainImage
    hybrid_recommendations = hybrid_recommender(target_product, target_image_path)
    return hybrid_recommendations