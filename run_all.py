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

TOPICS = ["business", "international business", "politics", "social or casual events", "sporting events"]
ACTOR_TYPES = ["allies", "neutral", "enemies"]
WORLD_TYPE = "real world"
PROMPTS_PER_COMBINATION = 5
OUTPUT_CSV = Path("results.csv")

CSV_FIELDS = [
    "game_type",
    "role",
    "framing_condition",
    "stake_level",
    "actor_type",
    "model_name",
    "prompt",
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

    # Match bare "A"/"B" or verbose "Decision A"/"Decision B"
    m = re.search(r'<decision>\s*(?:Decision\s*)?([AB])\s*(?:\s*\(.*?\))?\s*</decision>', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def _strip_llama_answer(text: str) -> str:
    """Remove everything from <justification> onward so decision models
    see only the vignette and the decision question, not Llama's answer."""
    cut = re.search(r'<justification>', text, re.IGNORECASE)
    if cut:
        return text[:cut.start()].rstrip()
    # fallback: cut at <decision> tag if no justification tag
    cut = re.search(r'<decision>', text, re.IGNORECASE)
    if cut:
        return text[:cut.start()].rstrip()
    return text


def _wrap_for_decision(text: str) -> str:
    """Wrap the vignette in a strict instruction that prevents the model from
    giving analysis instead of playing the role."""
    return (
        "Read the following scenario carefully and respond ONLY as the character "
        "described. Do NOT provide analysis, commentary, or a payoff matrix. "
        "You MUST end your response with a <decision> tag containing your choice.\n\n"
        + text
    )


async def collect_decisions(api_client: APIClient, story: object, game_type: str) -> dict:
    clean_content = _strip_llama_answer(story.content)
    wrapped = _wrap_for_decision(clean_content)
    responses = await api_client.generate(wrapped, model="all")
    return {
        model: (
            clean_content,  # store the clean story, not the wrapper
            responses.get(model),
            extract_decision(responses.get(model) or "", game_type)
        )
        for model in DECISION_MODELS
    }


def write_rows(writer: csv.DictWriter, decisions: dict, base_row: dict) -> None:
    for model, (prompt, raw_response, decision) in decisions.items():
        writer.writerow({
            **base_row,
            "model_name":   model,
            "prompt":       prompt,
            "raw_response": raw_response,
            "decision":     decision,
        })


async def run_pd(api_client: APIClient, writer: csv.DictWriter) -> None:
    seen: set = set()
    generator = PDStoryGenerator(api_client)
    for stakes, topic, actor_type in product(Stakes, TOPICS, ACTOR_TYPES):
        game = PrisonersDilemmaGame(stakes=stakes)
        logger.info(f"PD | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        unique_prompt = ""
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_pd(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                n_stories=1,
                unique_prompt=unique_prompt,
            )
            if not stories:
                logger.warning(f"PD | no story returned for iteration {i + 1}")
                continue
            if stories[0].content in seen:
                logger.warning(f"PD | duplicate story skipped for iteration {i + 1}")
                continue
            seen.add(stories[0].content)
            # Update unique_prompt so next iteration avoids the same characters/setting
            summary = await generator.generate_story_summary(stories[0].content)
            unique_prompt = generator.generate_unique_prompt(
                [s for s in seen][:i+1]  # pass summaries of seen stories
            )
            unique_prompt = generator.generate_unique_prompt([summary]) if not unique_prompt else unique_prompt
            decisions = await collect_decisions(api_client, stories[0], "prisoner's_dilemma")
            write_rows(writer, decisions, {
                "game_type":         "prisoner's_dilemma",
                "role":              "player",
                "framing_condition": topic,
                "stake_level":       get_stake_label(stakes),
                "actor_type":        actor_type,
            })


async def run_ug(api_client: APIClient, writer: csv.DictWriter) -> None:
    seen: set = set()
    generator = UltimatumStoryGenerator(api_client)
    for stakes, topic, actor_type, role in product(Stakes, TOPICS, ACTOR_TYPES, ["proposer", "responder"]):
        game = UltimatumGame(total_amount=20, stakes=stakes)
        logger.info(f"UG | role={role} | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        unique_prompt = ""
        summaries = []
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_ultimatum(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                role=role,
                n_stories=1,
                unique_prompt=unique_prompt,
            )
            if not stories:
                logger.warning(f"UG | no story returned for iteration {i + 1}")
                continue
            if stories[0].content in seen:
                logger.warning(f"UG | duplicate story skipped for iteration {i + 1}")
                continue
            seen.add(stories[0].content)
            summary = await generator.generate_story_summary(stories[0].content)
            summaries.append(summary)
            unique_prompt = generator.generate_unique_prompt(summaries)
            game_type_key = f"ultimatum_game_{role}"
            decisions = await collect_decisions(api_client, stories[0], game_type_key)
            write_rows(writer, decisions, {
                "game_type":         "ultimatum_game",
                "role":              role,
                "framing_condition": topic,
                "stake_level":       get_stake_label(stakes),
                "actor_type":        actor_type,
            })


async def run_pgg(api_client: APIClient, writer: csv.DictWriter) -> None:
    seen: set = set()
    generator = PGGStoryGenerator(api_client)
    for stakes, topic, actor_type in product(Stakes, TOPICS, ACTOR_TYPES):
        game = PublicGoodsGame(num_players=4, endowment=20, multiplier=1.6, stakes=stakes)
        logger.info(f"PGG | stakes={stakes.name} | topic={topic} | actor={actor_type}")
        unique_prompt = ""
        summaries = []
        for i in range(PROMPTS_PER_COMBINATION):
            stories = await generator.generate_stories_pgg(
                game=game,
                topic=topic,
                world_type=WORLD_TYPE,
                actor_type=actor_type,
                n_stories=1,
                unique_prompt=unique_prompt,
            )
            if not stories:
                logger.warning(f"PGG | no story returned for iteration {i + 1}")
                continue
            if stories[0].content in seen:
                logger.warning(f"PGG | duplicate story skipped for iteration {i + 1}")
                continue
            seen.add(stories[0].content)
            summary = await generator.generate_story_summary(stories[0].content)
            summaries.append(summary)
            unique_prompt = generator.generate_unique_prompt(summaries)
            decisions = await collect_decisions(api_client, stories[0], "public_goods_game")
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
        await run_ug(api_client, writer)
        await run_pgg(api_client, writer)

    logger.info(f"Done. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())