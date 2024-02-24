from fastapi import APIRouter
from pydantic import BaseModel

from .nlp.fasttext_unit import get_most_similar_top_nine

router = APIRouter(prefix="/ai", tags=["ai"])

class TxtItem(BaseModel):
    content: str

@router.post("/txt-to-similarity")
async def calculate_img_to_similarity(item: TxtItem):
    final_result = get_most_similar_top_nine(item.content.replace(", ", ",").split(","))
    return final_result