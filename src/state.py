from typing import TypedDict, Dict, Any, Optional

class GardAgentState(TypedDict):
    # Data Masuk dari Platform (Express/Flutter)
    user_message: str              
    image_path: Optional[str]      
    baseline_gerdq: str            # "RENDAH", "SEDANG", "TINGGI"
    vitals_data: Dict[str, Any]    # Smartwatch JSON
    activity_data: Dict[str, Any]  # Google Calendar JSON
    
    # Komunikasi Antar Ejen
    input_route: str               
    food_analysis: Dict[str, Any]  
    lifestyle_analysis: Dict[str, Any] 
    
    # Hasil Akhir
    risk_score: int                
    medical_context: str           
    final_response: str