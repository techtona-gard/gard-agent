from typing import Dict, Any

from src.state import GardAgentState


def nutri_scan_node(state: GardAgentState) -> Dict[str, Any]:
    print("\n🥗 Nutri Scan Agent dijalankan")

    return {
        "food_analysis": {
            "status": "ok"
        }
    }