from typing import Dict, Any

from src.state import GardAgentState


def gerd_expert_node(state: GardAgentState) -> Dict[str, Any]:
    print("\n🩺 GERD Expert Agent dijalankan")

    return {
        "final_response": "Analisis Berhasil!"
    }