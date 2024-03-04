import base64
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from fastapi.routing import APIRouter
import torchvision.transforms as transforms
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

from .model.HybridModel import hybrid_recommender

from .model.ImageSearch import image_processing_and_search
from .model.AgeGenderModel import recommend_by_age_gender

router = APIRouter(prefix="/ai", tags=["ai"])

class TxtItem(BaseModel):
    content: str

class ProductItem(BaseModel):
    productName: str
    productMainImage: str

class MemberInfo(BaseModel):
    ageGroup: int
    gender: int

@router.post("/hybrid-recommendations")
async def calculate_recommendations_scores(item: ProductItem):
    target_product = item.productName
    target_image_path = item.productMainImage
    hybrid_recommendations = hybrid_recommender(target_product, target_image_path)
    return hybrid_recommendations

@router.post("/image-search")
async def image_search(file: bytes = File(...)):
    img = Image.open(BytesIO(base64.b64decode(file)))
    image_search_results = image_processing_and_search(img)
    return image_search_results

@router.post("/age-gender-recommendation")
async def age_gender_recommendation(info: MemberInfo):
    target_age = info.ageGroup
    target_gender = info.gender
    print(target_age, target_gender)
    recommendations = recommend_by_age_gender(target_age, target_gender)
    return recommendations
