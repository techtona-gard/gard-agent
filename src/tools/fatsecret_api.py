import httpx
from typing import Optional
from src.schemas import FoodProfile

class FatSecretAPI:
    def __init__(self):
        self.base_url = "https://dummy-fatsecret-api.com"
        
    def get_nutrition(self, food_name: str) -> FoodProfile:
        """
        Hit API FatSecret via requests.
        Dummy implementation for now.
        """
        # Simulasi deteksi makanan pemicu GERD
        trigger_foods = ["cokelat", "pedas", "kopi", "asam", "gorengan", "santan", "soda"]
        is_trigger = any(t in food_name.lower() for t in trigger_foods)
        
        return FoodProfile(
            food_name=food_name,
            calories=250.0,
            fat=15.0 if is_trigger else 5.0,
            protein=10.0,
            carbs=20.0,
            is_gerd_trigger=is_trigger
        )

fatsecret_tool = FatSecretAPI()
