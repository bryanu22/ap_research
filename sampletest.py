import asyncio
import csv
import logging
import re
from itertools import product
from pathlib import Path

from game_theory_llm import (
    PrisonersDilemmaGame,
    UltimatumGame,
    PublicGoodsGame,
    PDStoryGenerator,
    UltimatumStoryGenerator,
    PGGStoryGenerator,
    APIClient,
    Stakes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPICS = ["business"]
ACTOR_TYPES = ["enemies"]
WORLD_TYPE = "real world"
PROMPTS_PER_COMBINATION = 2
OUTPUT_CSV = Path("prelim_results.csv")

CSV_FIELDS = [
    "game_type",
    "role",
    "framing_condition",
    "stake_level",
    "actor_type",
    "model_name",
    "raw_response",
    "decision",
]

DECISION_MODELS = ["claude", "gpt4"]


def get_stake_label(stakes: Stakes) -> str:
    return {Stakes.LOW: "low", Stakes.MEDIUM: "medium", Stakes.HIGH: "high"}[stakes]


def extract_decision(text: str, game_type: str) -> str:
    if game_type == "ultimatum_game_responder":
        m = re.search(r'<decision>\s*(accept|reject)\s*</decision>', text, re.IGNORECASE)
        return m.group(1).lower() if m else None

    m = re.search(r'<decision>\s*\$?([0-9]+)\s*</decision>', text, re.IGNORECASE)
    if m and game_type in ("ultimatum_game_proposer", "public_goods_game"):
        return m.group(1)

    m = re.search(r'<decision>\s*([AB])\s*</decision>', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


async def collect_decisions(api_client: APIClient, story_content: str, game_type: str) -> dict:
    responses = await api_client.generate(story_content, model="all")
    return {
        model: (
            responses.get(model),
            extract_decision(responses.get(model) or "", game_type)
        )
        for model in DECISION_MODELS
    }


def write_rows(writer: csv.DictWriter, decisions: dict, base_row: dict) -> None:
    for model, (raw_response, decision) in decisions.items():
        writer.writerow({
            **base_row,
            "model_name":   model,
            "raw_response": raw_response,
            "decision":     decision,
        })


async def run_pd(api_client: APIClient, writer: csv.DictWriter) -> None:
    generator = PDStoryGenerator(api_client)
    for stakes, topic, actor_type in product(Stakes, TOPICS, ACTOR_TYPES):
        game = PrisonersDilemmaGame(stakes=stakes)
        logger.info(f"PD | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_pd(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                n_stories=1,
            )
            if not stories:
                logger.warning(f"PD | no story returned for iteration {i + 1}")
                continue
            decisions = await collect_decisions(api_client, stories[0].content, "prisoner's_dilemma")
            write_rows(writer, decisions, {
                "game_type":         "prisoner's_dilemma",
                "role":              "player",
                "framing_condition": topic,
                "stake_level":       get_stake_label(stakes),
                "actor_type":        actor_type,
            })


async def run_ug(api_client: APIClient, writer: csv.DictWriter) -> None:
    generator = UltimatumStoryGenerator(api_client)
    for stakes, topic, actor_type, role in product(Stakes, TOPICS, ACTOR_TYPES, ["proposer", "responder"]):
        game = UltimatumGame(total_amount=20, stakes=stakes)
        logger.info(f"UG | role={role} | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_ultimatum(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                role=role,
                n_stories=1,
            )
            if not stories:
                logger.warning(f"UG | no story returned for iteration {i + 1}")
                continue
            game_type_key = f"ultimatum_game_{role}"
            decisions = await collect_decisions(api_client, stories[0].content, game_type_key)
            write_rows(writer, decisions, {
                "game_type":         "ultimatum_game",
                "role":              role,
                "framing_condition": topic,
                "stake_level":       get_stake_label(stakes),
                "actor_type":        actor_type,
            })


async def run_pgg(api_client: APIClient, writer: csv.DictWriter) -> None:
    generator = PGGStoryGenerator(api_client)
    for stakes, topic, actor_type in product(Stakes, TOPICS, ACTOR_TYPES):
        game = PublicGoodsGame(num_players=4, endowment=20, multiplier=1.6, stakes=stakes)
        logger.info(f"PGG | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_pgg(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                n_stories=1,
            )
            if not stories:
                logger.warning(f"PGG | no story returned for iteration {i + 1}")
                continue
            decisions = await collect_decisions(api_client, stories[0].content, "public_goods_game")
            write_rows(writer, decisions, {
                "game_type":         "public_goods_game",
                "role":              "contributor",
                "framing_condition": topic,
                "stake_level":       get_stake_label(stakes),
                "actor_type":        actor_type,
            })


async def main() -> None:
    api_client = APIClient()

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        await run_pd(api_client, writer)
        # await run_ug(api_client, writer)
        # await run_pgg(api_client, writer)

    logger.info(f"Done. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())