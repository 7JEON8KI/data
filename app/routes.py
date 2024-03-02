from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
from fastapi.routing import APIRouter
import torchvision.transforms as transforms
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

from .model.HybridModel import hybrid_recommender

from .model.ImageSearch import image_processing_and_search

router = APIRouter(prefix="/ai", tags=["ai"])

class TxtItem(BaseModel):
    content: str

class ProductItem(BaseModel):
    productName: str
    productMainImage: str

# class SearchImage(BaseModel):
#     productImage: UploadFile

@router.post("/hybrid-recommendations")
async def calculate_recommendations_scores(item: ProductItem):
    target_product = item.productName
    target_image_path = item.productMainImage
    hybrid_recommendations = hybrid_recommender(target_product, target_image_path)
    return hybrid_recommendations

@router.post("/image-search")
async def image_search(file: UploadFile):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    image_search_results = image_processing_and_search(img)
    return image_search_results