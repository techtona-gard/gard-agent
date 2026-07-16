from typing import Dict, Any

from src.state import GardAgentState


def bio_rhythm_node(state: GardAgentState) -> Dict[str, Any]:
    print("\n🛌 Bio Rhythm Agent dijalankan")

    return {
        "lifestyle_analysis": {
            "status": "ok"
        }
    }