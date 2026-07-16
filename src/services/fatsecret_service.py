import os
import time
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Standard Indonesian / Student food mock DB fallback
MOCK_FOOD_DB = {
    "indomie": {"calories": 380, "fat": 14.0, "spicy": False, "acidic": False, "caffeine": False},
    "mie instan": {"calories": 380, "fat": 14.0, "spicy": False, "acidic": False, "caffeine": False},
    "mie goreng pedas": {"calories": 400, "fat": 16.0, "spicy": True, "acidic": False, "caffeine": False},
    "ayam goreng": {"calories": 260, "fat": 15.0, "spicy": False, "acidic": False, "caffeine": False},
    "kopi hitam": {"calories": 2, "fat": 0.0, "spicy": False, "acidic": True, "caffeine": True},
    "kopi susu": {"calories": 120, "fat": 4.5, "spicy": False, "acidic": True, "caffeine": True},
    "jus jeruk": {"calories": 110, "fat": 0.2, "spicy": False, "acidic": True, "caffeine": False},
    "pisang": {"calories": 105, "fat": 0.3, "spicy": False, "acidic": False, "caffeine": False},
    "susu sapi": {"calories": 150, "fat": 8.0, "spicy": False, "acidic": False, "caffeine": False},
    "cokelat": {"calories": 220, "fat": 12.0, "spicy": False, "acidic": False, "caffeine": False, "chocolate": True},
    "teh": {"calories": 2, "fat": 0.0, "spicy": False, "acidic": False, "caffeine": True},
}

class FatSecretService:
    def __init__(self):
        self.client_id = os.getenv("FATSECRET_CLIENT_ID")
        self.client_secret = os.getenv("FATSECRET_CLIENT_SECRET")
        self.token_url = "https://oauth.fatsecret.com/connect/token"
        self.api_url = "https://platform.fatsecret.com/rest/server.api"
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def _get_token(self) -> Optional[str]:
        """
        Gets OAuth2 token via client credentials. Cached in memory.
        """
        if not self.client_id or not self.client_secret:
            logger.warning("FatSecret Service: Credentials missing. Using mock fallback mode.")
            return None

        # Return cached token if valid
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        try:
            response = requests.post(
                self.token_url,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials", "scope": "basic"},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            self._access_token = data.get("access_token")
            # Set expiry with a safety margin of 60 seconds
            self._token_expires_at = time.time() + float(data.get("expires_in", 3600)) - 60
            logger.info("FatSecret Service: Successfully authenticated OAuth2 token.")
            return self._access_token
        except Exception as e:
            logger.error(f"FatSecret Service: OAuth2 Authentication failed: {str(e)}")
            return None

    def search_food_nutrition(self, food_name: str) -> Dict[str, Any]:
        """
        Search for a food item and return nutrition information.
        Falls back to local mock database if credentials aren't set or API fails.
        """
        logger.info(f"FatSecret Service: Querying nutrition for '{food_name}'")
        
        token = self._get_token()
        if token:
            try:
                # 1. Search for food ID
                search_params = {
                    "method": "foods.search",
                    "search_expression": food_name,
                    "format": "json"
                }
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(self.api_url, params=search_params, headers=headers, timeout=5)
                response.raise_for_status()
                search_results = response.json()
                
                food_list = search_results.get("foods", {}).get("food", [])
                if food_list:
                    # Get details of the first search result
                    food_id = food_list[0].get("food_id") if isinstance(food_list, list) else food_list.get("food_id")
                    
                    if food_id:
                        detail_params = {
                            "method": "food.get.v2",
                            "food_id": food_id,
                            "format": "json"
                        }
                        detail_res = requests.get(self.api_url, params=detail_params, headers=headers, timeout=5)
                        detail_res.raise_for_status()
                        detail_data = detail_res.json()
                        
                        # Parse nutrition facts
                        servings = detail_data.get("food", {}).get("servings", {}).get("serving", [])
                        serving = servings[0] if isinstance(servings, list) else servings
                        
                        fat = float(serving.get("fat", 0))
                        calories = float(serving.get("calories", 0))
                        
                        # Infer spices/acidity based on title
                        title_lower = food_name.lower()
                        is_spicy = any(w in title_lower for w in ["pedas", "cabai", "rica", "balado", "sambal", "spicy", "chili"])
                        is_acidic = any(w in title_lower for w in ["jeruk", "orange", "lemon", "tomat", "acid", "cuka", "vinegar"])
                        is_caffeine = any(w in title_lower for w in ["kopi", "coffee", "teh", "tea", "energy", "caffeine"])
                        
                        return {
                            "calories": calories,
                            "fat": fat,
                            "spicy": is_spicy,
                            "acidic": is_acidic,
                            "caffeine": is_caffeine
                        }
            except Exception as e:
                logger.error(f"FatSecret Service: API query failed, falling back to mock: {str(e)}")

        # Fallback to Mock Database matching
        query = food_name.lower()
        for key, mock_data in MOCK_FOOD_DB.items():
            if key in query:
                logger.info(f"FatSecret Service: Mock database matched '{key}' for '{food_name}'")
                return mock_data
                
        # Default fallback values
        logger.warning(f"FatSecret Service: No match found for '{food_name}'. Using default safe profiles.")
        return {
            "calories": 150,
            "fat": 5.0,
            "spicy": False,
            "acidic": False,
            "caffeine": False
        }
