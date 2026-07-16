from enum import Enum


class Route(str, Enum):
    PROCESS_FOOD = "PROCESS_FOOD"
    PROCESS_LIFESTYLE = "PROCESS_LIFESTYLE"
    GENERAL_CHAT = "GENERAL_CHAT"