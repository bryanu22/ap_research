# game_theory_llm/models.py
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class Game(ABC):
    name: str
    num_players: int

    @abstractmethod
    def get_actions(self, player_id: int):
        pass

    @abstractmethod
    def compute_payoffs(self, actions: Dict[int, Any]):
        pass

@dataclass
class PayoffMatrix:
    """Represents a 2x2 payoff matrix for game theory scenarios."""
    matrix: List[Tuple[int, int]]
    
    def __post_init__(self):
        if len(self.matrix) != 4:
            raise ValueError("Payoff matrix must contain exactly 4 scenarios")
    
    def format_matrix(self) -> str:
        """Returns a formatted string representation of the payoff matrix."""
        return f"""
┌──────────────┬─────────────┬─────────────┐
│ Agent 1 ↓    │  Agent 2 →  │             │
├──────────────┼─────────────┼─────────────┤
│              │      A      │      B      │
├──────────────┼─────────────┼─────────────┤
│      A       │  {self.matrix[0][0]}, {self.matrix[0][1]}  │  {self.matrix[1][0]}, {self.matrix[1][1]}  │
├──────────────┼─────────────┼─────────────┤
│      B       │  {self.matrix[2][0]}, {self.matrix[2][1]}  │  {self.matrix[3][0]}, {self.matrix[3][1]}  │
└──────────────┴─────────────┴─────────────┘
"""

@dataclass
class UltimatumGame(Game):
    total_amount: int = 100

    name: str = "Ultimatum Game"
    num_players: int = 2

    def get_actions(self, player_id: int):
        if player_id == 0:
            return list(range(self.total_amount + 1)) 
        else:
            return ["accept", "reject"]

    def compute_payoffs(self, actions: Dict[int, Any]):
        offer = actions[0]
        response = actions[1]

        if response == "accept":
            return {
                0: self.total_amount - offer,
                1: offer,
            }
        else:
            return {
                0: 0,
                1: 0,
            }

@dataclass
class Story:
    """A generated story and its metadata."""
    content: str
    topic: str
    world_type: str
    actor_type: str
    prompt: str = None
    decision: str = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()