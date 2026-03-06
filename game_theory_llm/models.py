from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum


class Stakes(Enum):
    LOW    = 1
    MEDIUM = 5
    HIGH   = 25


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
    matrix: List[Tuple[int, int]]

    def __post_init__(self):
        if len(self.matrix) != 4:
            raise ValueError("Payoff matrix must contain exactly 4 scenarios")

    def format_matrix(self) -> str:
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
class PrisonersDilemmaGame(Game):
    stakes: Stakes = Stakes.LOW

    BASE_REWARD: int = field(default=4, init=False, repr=False)
    name: str = field(default="Prisoner's Dilemma", init=False, repr=False)
    num_players: int = field(default=2, init=False, repr=False)

    @property
    def reward(self) -> int:
        return self.BASE_REWARD * self.stakes.value

    @property
    def temptation(self) -> int:
        return int(self.reward * 1.5)

    @property
    def punishment(self) -> int:
        return int(self.reward * 0.5)

    @property
    def sucker(self) -> int:
        return 0

    @property
    def payoff_matrix(self) -> PayoffMatrix:
        R = self.reward
        T = self.temptation
        P = self.punishment
        S = self.sucker
        return PayoffMatrix([
            (R, R),
            (S, T),
            (T, S),
            (P, P),
        ])

    def get_actions(self, player_id: int):
        return ["cooperate", "defect"]

    def compute_payoffs(self, actions: Dict[int, Any]) -> Dict[int, int]:
        a0 = actions[0]
        a1 = actions[1]
        R, T, P, S = self.reward, self.temptation, self.punishment, self.sucker
        if a0 == "cooperate" and a1 == "cooperate":
            return {0: R, 1: R}
        if a0 == "cooperate" and a1 == "defect":
            return {0: S, 1: T}
        if a0 == "defect" and a1 == "cooperate":
            return {0: T, 1: S}
        return {0: P, 1: P}


@dataclass
class UltimatumGame(Game):
    total_amount: int = 20
    stakes: Stakes = Stakes.LOW

    name: str = field(default="Ultimatum Game", init=False, repr=False)
    num_players: int = field(default=2, init=False, repr=False)

    @property
    def effective_total(self) -> int:
        return self.total_amount * self.stakes.value

    def get_actions(self, player_id: int):
        if player_id == 0:
            return list(range(self.effective_total + 1))
        return ["accept", "reject"]

    def compute_payoffs(self, actions: Dict[int, Any]) -> Dict[int, int]:
        offer    = actions[0]
        response = actions[1]
        if response == "accept":
            return {0: self.effective_total - offer, 1: offer}
        return {0: 0, 1: 0}


@dataclass
class PublicGoodsGame(Game):
    num_players: int = 4
    endowment: int = 20
    multiplier: float = 1.6
    stakes: Stakes = Stakes.LOW

    name: str = field(default="Public Goods Game", init=False, repr=False)

    @property
    def effective_endowment(self) -> int:
        return self.endowment * self.stakes.value

    def get_actions(self, player_id: int):
        return list(range(self.effective_endowment + 1))

    def compute_payoffs(self, actions: Dict[int, Any]) -> Dict[int, float]:
        total_contribution = sum(actions.values())
        pool_value         = total_contribution * self.multiplier
        per_player_share   = pool_value / self.num_players
        return {
            player_id: self.effective_endowment - contribution + per_player_share
            for player_id, contribution in actions.items()
        }


@dataclass
class Story:
    content: str
    topic: str
    world_type: str
    actor_type: str
    prompt: Optional[str] = None
    decision: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()