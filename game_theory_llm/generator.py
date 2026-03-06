from typing import List, Tuple, Optional, Literal
import re
import logging
from .models import PayoffMatrix, Story, UltimatumGame, PublicGoodsGame, PrisonersDilemmaGame, Stakes
from .api import APIClient
from .config import WORLD_DICT, ACTOR_TYPES, TOPICS
import asyncio
from dataclasses import dataclass
import textwrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UGRole = Literal["proposer", "responder"]

_STAKE_HOURS = {
    Stakes.LOW:    2.5,
    Stakes.MEDIUM: 12.5,
    Stakes.HIGH:   62.5,
}


def _stakes_instruction(stakes: Stakes) -> str:
    """Return a prompt instruction that tells Llama how large the stakes should be
    in terms of hours of wages, without prescribing a dollar amount.
    Llama picks the wage rate appropriate to the characters it writes."""
    hours = _STAKE_HOURS[stakes]
    return (
        f"The amounts or resources at stake in this scenario should be equivalent to "
        f"approximately {hours} hours of wages for the type of person you are writing about. "
        f"Choose an hourly wage rate appropriate to the characters and context "
        f"(e.g. a senior professional earns more per hour than a casual worker), "
        f"then set the actual amount accordingly. "
        f"The stake does not have to be money — it can be any resource, favour, or "
        f"asset that makes sense in context, as long as you express it as a concrete "
        f"number (e.g. '$120', '15 hours of work', '3 tickets worth $40 each')."
    )


@dataclass
class BatchGenerationResult:
    stories: List[Story]
    summaries: List[str]
    unique_prompt: str


class StoryGenerator:

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("StoryGenerator initialized")

    async def generate_story_summary(self, story: str) -> str:
        logger.debug("Generating summary for story")
        prompt = """Analyze this story and create a single, comprehensive sentence that captures:
        1. The main character(s) and their defining traits
        2. The primary setting/location
        3. The core conflict or goal
        4. The most important plot development

        Story to summarize:
        {story}"""

        try:
            summary = await self.api_client.generate(prompt.format(story=story), model="llama")
            summary = summary["llama"]
            logger.debug(f"Generated summary: {summary[:100]}...")
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"A story about {story[:100]}..."

    def generate_unique_prompt(self, summaries: List[str]) -> str:
        logger.debug(f"Generating unique prompt from {len(summaries)} summaries")
        if not summaries:
            return ""
        all_summaries = "\n".join(f"{i+1}. {summary}" for i, summary in enumerate(summaries))
        return f"""Previous story summaries (avoid reusing elements from these):
        {all_summaries}

        Generate completely new stories with different characters, settings, and plots."""

    def create_query(
        self,
        matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> str:
        logger.debug("Creating query prompt")
        prompt = f"""Write {number_of_stories} unique stories about a scenario involving two agents and their possible actions.
        This matrix {matrix.format_matrix()} represents each agent's happiness based on their decision and the other agent's decision.
        The topic you need to write about is {topic}.
        The relationship between the two agents is {world_type} {actor_type}.

        Please write {number_of_stories} stories that would present this situation as a word problem having to do with {topic} without making it obvious that this is based on a game theory problem. Be creative and varied in your story structures and motifs. The relationship between the agents should be that they are {world_type} {actor_type}.

        It should be clear that each agent has two possible choices, which should be labeled as Decision A and Decision B in the story. Construct the narratives so that the potential outcomes for each agent's happiness align with the matrix below.
        - If both agents make decision A, then agent 1 will have happiness {matrix.matrix[0][0]} and agent 2 will have happiness {matrix.matrix[0][1]}.
        - If agent 1 makes decision A and agent 2 makes decision B then agent 1 will have happiness {matrix.matrix[1][0]} and agent 2 will have happiness {matrix.matrix[1][1]}.
        - If agent 2 makes decision A and agent 1 makes decision B then agent 1 will have happiness {matrix.matrix[2][0]} and agent 2 will have happiness {matrix.matrix[2][1]}.
        - If both agents make decision B then agent 1 will have happiness {matrix.matrix[3][0]} and agent 2 will have happiness {matrix.matrix[3][1]}.

        {WORLD_DICT[world_type]}

        RELATIONSHIP DESCRIPTION:
        {ACTOR_TYPES[actor_type]['description']}

        EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
        {ACTOR_TYPES[actor_type]['types']}

        Rules:
        1. Do not under any circumstance mention that this is a game
        2. You must explicitly label decisions as Decision A or Decision B
        3. Don't show explicit payoffs
        4. Show how outcomes depend on both agents' decisions
        5. Surround each story with <story></story> tags

        {unique_prompt}

        End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with one of the agents in the story.
        <toadd>
        You are <agent> in this scenario. What decision will you make?
        Provide your justification for your decision inside <justification></justification> tags.
        Then, output your decision, either: <decision>B</decision> or <decision>A</decision>. Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order.
        </toadd>"""

        return textwrap.dedent(prompt)

    async def generate_batch(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> BatchGenerationResult:
        logger.info("Generating batch of stories")
        prompt = self.create_query(
            payoff_matrix, topic, world_type, actor_type, unique_prompt, number_of_stories
        )

        try:
            content = await self.api_client.generate(prompt, model="llama")
            content = content["llama"]

            raw_stories = re.findall(r'<story>(.*?)</story>', content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories from response")

            if not raw_stories:
                logger.warning("No stories found in generated content")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                decision = self._extract_decision(story_content)
                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision,
                )
                stories.append(story)

            summaries = await asyncio.gather(
                *[self.generate_story_summary(story.content) for story in stories]
            )
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"Successfully generated batch with {len(stories)} stories")
            return BatchGenerationResult(stories, summaries, new_unique_prompt)

        except Exception as e:
            logger.error(f"Error generating batch: {str(e)}")
            return BatchGenerationResult([], [], "")

    async def generate_stories(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        n_stories: int = 100,
        batch_size: int = 10,
    ) -> List[Story]:
        logger.info(f"Starting generation of {n_stories} stories in batches of {batch_size}")

        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories: List[Story] = []
        all_summaries: List[str] = []
        unique_prompt = ""
        n_batches = (n_stories + batch_size - 1) // batch_size
        number_of_stories = n_stories if n_stories < 10 else 10

        for batch_num in range(n_batches):
            logger.info(f"Generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch(
                payoff_matrix, topic, world_type, actor_type, unique_prompt, number_of_stories,
            )

            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                all_summaries.extend(batch_result.summaries)
                unique_prompt = batch_result.unique_prompt
            else:
                logger.warning(f"Batch {batch_num + 1} generated no stories")

            if len(all_stories) >= n_stories:
                break

        logger.info(f"Generation complete. Generated {len(all_stories)} stories total")
        return all_stories[:n_stories]

    def _extract_decision(self, text: str) -> Optional[str]:
        patterns = [
            (r'<decision>\s*([AB])\s*</decision>',        {'A': 'A', 'B': 'B'}),
            (r'<decision>\s*([12])\s*</decision>',         {'1': 'A', '2': 'B'}),
            (r'<decision>\s*(\&{1,2})\s*</decision>',      {'&': '&', '&&': '&&'}),
            (r'<decision>\s*(yellow|green)\s*</decision>', {'yellow': 'A', 'green': 'B'}),
        ]
        for pattern, mapping in patterns:
            match = re.search(pattern, text)
            if match:
                decision = match.group(1)
                mapped_decision = mapping.get(decision)
                if mapped_decision:
                    logger.debug(f"Extracted decision: {decision} -> mapped to: {mapped_decision}")
                    return mapped_decision
        logger.warning("No valid decision pattern found in story")
        return None


class PDStoryGenerator(StoryGenerator):

    def create_query_pd(
        self,
        game: PrisonersDilemmaGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> str:
        logger.debug("Creating PD query prompt")
        R  = game.reward
        T  = game.temptation
        P  = game.punishment
        S  = game.sucker
        stakes_instr = _stakes_instruction(game.stakes)
        prompt = f"""Write {number_of_stories} unique stories about a scenario involving two individual people and their possible actions.
        The topic domain is {topic}.
        The relationship between the two people is {actor_type}.

        CRITICAL CASTING RULE: Both agents MUST be individual human beings — e.g., two colleagues,
        two friends, two competitors, two teammates, two traders, two athletes, two neighbors.
        Do NOT use corporations, governments, countries, or any non-human entities as agents.
        STAKES GUIDANCE: {stakes_instr}

        Each person independently chooses Decision A (cooperate) or Decision B (defect).
        Do NOT state explicit dollar amounts anywhere in the story body. Convey the payoff
        structure through narrative consequences only (e.g., one person ends up ahead,
        both benefit modestly, both walk away with less than they could have had).
        The underlying payoff structure is:
        - Both choose A: each ends up equally well off (mutual benefit).
        - Agent 1 chooses A, agent 2 chooses B: agent 1 ends up worst off, agent 2 ends up best off.
        - Agent 1 chooses B, agent 2 chooses A: agent 1 ends up best off, agent 2 ends up worst off.
        - Both choose B: each ends up worse than if both had chosen A (mutual loss relative to cooperation).

        TOPIC GUIDANCE — use {topic} to set the scene and relationship type:
        {ACTOR_TYPES[actor_type]['description']}

        EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
        {ACTOR_TYPES[actor_type]['types']}

        Rules:
        1. Do not under any circumstance mention that this is a game
        2. You must explicitly label decisions as Decision A or Decision B
        3. Convey outcome differences through narrative consequences
        4. Show how outcomes depend on both agents' decisions
        5. Surround each story with <story></story> tags

        {unique_prompt}

        End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with one of the people in the story.
        <toadd>
        You are <agent> in this scenario. What decision will you make?
        Provide your justification for your decision inside <justification></justification> tags.
        Then, output your decision, either: <decision>B</decision> or <decision>A</decision>. Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order.
        </toadd>"""

        return textwrap.dedent(prompt)

    async def generate_batch_pd(
        self,
        game: PrisonersDilemmaGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> BatchGenerationResult:
        logger.info(f"Generating PD batch — stakes={game.stakes.name}")
        prompt = self.create_query_pd(
            game, topic, world_type, actor_type, unique_prompt, number_of_stories
        )

        try:
            content = await self.api_client.generate(prompt, model="llama")
            content = content["llama"]

            raw_stories = re.findall(r'<story>(.*?)</story>', content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} PD stories")

            if not raw_stories:
                logger.warning("No PD stories found")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                decision = self._extract_decision(story_content)
                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision,
                )
                stories.append(story)

            summaries = await asyncio.gather(
                *[self.generate_story_summary(s.content) for s in stories]
            )
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"PD batch created {len(stories)} stories")
            return BatchGenerationResult(stories, summaries, new_unique_prompt)

        except Exception as e:
            logger.error(f"Error generating PD batch: {e}")
            return BatchGenerationResult([], [], "")

    async def generate_stories_pd(
        self,
        game: PrisonersDilemmaGame,
        topic: str,
        world_type: str,
        actor_type: str,
        n_stories: int = 100,
        batch_size: int = 10,
        unique_prompt: str = "",
    ) -> List[Story]:
        logger.info(f"Starting PD generation — stakes={game.stakes.name}, n={n_stories}")

        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories: List[Story] = []
        n_batches = (n_stories + batch_size - 1) // batch_size
        number_of_stories = n_stories if n_stories < 10 else 10

        for batch_num in range(n_batches):
            logger.info(f"PD generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch_pd(
                game, topic, world_type, actor_type, unique_prompt, number_of_stories
            )

            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                unique_prompt = batch_result.unique_prompt
            else:
                logger.warning(f"PD batch {batch_num + 1} produced no stories")

            if len(all_stories) >= n_stories:
                break

        logger.info(f"PD generation complete. Produced {len(all_stories)} stories")
        return all_stories[:n_stories]


class UltimatumStoryGenerator(StoryGenerator):

    def create_query_proposer(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> str:
        logger.debug("Creating UG proposer query prompt")
        total   = game.effective_total
        stakes_instr = _stakes_instruction(game.stakes)

        prompt = f"""Write {number_of_stories} unique short stories involving two agents:
one makes a monetary offer (the proposer) and the other accepts or rejects (the responder).
Do not call this a game. Keep it natural and varied.

Context:
- Total amount to split: {stakes_instr}
  Express this as a concrete number in your story. The proposer then chooses how much
  of that total to offer to the responder (any amount from $0 to the full total).
- The proposer keeps the remainder (${total} minus the offer).
- The responder will either accept or reject the offer.
- If the responder rejects, both get $0.

Topic domain: {topic}
Relationship: {actor_type}

CRITICAL CASTING RULE: Both agents MUST be individual human beings. Do NOT use corporations,
governments, or countries. Choose a dollar amount that is realistic and meaningful at an
individual human scale for the scenario — casting large organizations would make the stakes
implausible 
RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}

Rules for generation:
1. Surround each story with <story>...</story> tags.
2. Dollar amounts MAY appear in the question stem but not in the story narrative itself.
3. Vary names, settings, and motives. Avoid reusing elements from the provided unique prompt.
4. Make it clear which individual is the proposer and which is the responder.

{unique_prompt}

End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with the PROPOSER agent in the story:
<toadd>
You are <agent> in this scenario. You have ${total} to split. How much will you offer to the responder?
Provide your justification inside <justification></justification> tags.
Then output your offer as an integer: <decision>integer between 0 and {total}</decision>
</toadd>

Example story ending:
<justification>Offering just over half feels fair and avoids the risk of rejection.</justification>
<decision>55</decision>
"""
        return textwrap.dedent(prompt)

    def create_query_responder(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> str:
        logger.debug("Creating UG responder query prompt")
        total   = game.effective_total
        stakes_instr = _stakes_instruction(game.stakes)

        prompt = f"""Write {number_of_stories} unique short stories involving two agents:
one makes a monetary offer (the proposer) and the other accepts or rejects (the responder).
Do not call this a game. Keep it natural and varied.

Context:
- Total amount to split: {stakes_instr}
  Express this as a concrete number in your story.
- The proposer has already decided on an offer to give the responder.
- The responder can accept (both keep their shares) or reject (both receive nothing).

Topic domain: {topic}
Relationship: {actor_type}

CRITICAL CASTING RULE: Both agents MUST be individual human beings. Do NOT use corporations,
governments, or countries. Choose a dollar amount that is realistic and meaningful at an
individual human scale for the scenario — casting large organizations would make the stakes
implausible

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}

Rules for generation:
1. Surround each story with <story>...</story> tags.
2. Embed a specific offer amount in the narrative (use <offer>integer</offer> to tag it).
3. Dollar amounts MAY appear in the question stem but keep narrative consequences qualitative.
4. Vary names, settings, offer amounts, and motives. Avoid reusing elements from the provided unique prompt.
5. Make it clear which individual is the proposer and which is the responder.

{unique_prompt}

End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with the RESPONDER agent in the story:
<toadd>
You are <agent> in this scenario. The proposer has offered you <offer>$X</offer> out of ${total}. Do you accept or reject?
Provide your justification inside <justification></justification> tags.
Then output your decision: <decision>accept</decision> or <decision>reject</decision>
</toadd>

Example story ending:
<offer>40</offer>
<justification>The offer is less than half but still meaningful given the circumstances.</justification>
<decision>accept</decision>
"""
        return textwrap.dedent(prompt)

    async def generate_batch_ultimatum(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        role: UGRole = "responder",
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> BatchGenerationResult:
        logger.info(f"Generating UG batch — role={role}")

        prompt = (
            self.create_query_proposer(game, topic, world_type, actor_type, unique_prompt, number_of_stories)
            if role == "proposer"
            else self.create_query_responder(game, topic, world_type, actor_type, unique_prompt, number_of_stories)
        )

        try:
            resp = await self.api_client.generate(prompt, model="llama")
            content = resp["llama"] if isinstance(resp, dict) and "llama" in resp else resp
            if isinstance(content, dict):
                content = str(content)

            raw_stories = re.findall(r'<story>(.*?)</story>', content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} UG stories (role={role})")

            if not raw_stories:
                logger.warning("No UG stories found")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                if role == "proposer":
                    decision_field = self._extract_proposer_offer(story_content, game.effective_total)
                else:
                    _, response    = self._extract_ultimatum(story_content, game.effective_total)
                    decision_field = response

                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision_field,
                )
                stories.append(story)

            summaries = await asyncio.gather(
                *[self.generate_story_summary(s.content) for s in stories]
            )
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"UG batch created {len(stories)} stories (role={role})")
            return BatchGenerationResult(stories, summaries, new_unique_prompt)

        except Exception as e:
            logger.error(f"Error generating UG batch: {e}")
            return BatchGenerationResult([], [], "")

    async def generate_stories_ultimatum(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        role: UGRole = "responder",
        n_stories: int = 100,
        batch_size: int = 10,
        unique_prompt: str = "",
    ) -> List[Story]:
        logger.info(f"Starting UG generation — role={role}, n={n_stories}")

        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")
        if role not in ("proposer", "responder"):
            raise ValueError("role must be 'proposer' or 'responder'")

        all_stories: List[Story] = []
        n_batches = (n_stories + batch_size - 1) // batch_size
        number_of_stories = n_stories if n_stories < 10 else 10

        for batch_num in range(n_batches):
            logger.info(f"UG generating batch {batch_num + 1}/{n_batches} (role={role})")
            batch_result = await self.generate_batch_ultimatum(
                game, topic, world_type, actor_type,
                role=role,
                unique_prompt=unique_prompt,
                number_of_stories=number_of_stories,
            )

            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                unique_prompt = batch_result.unique_prompt
            else:
                logger.warning(f"UG batch {batch_num + 1} produced no stories (role={role})")

            if len(all_stories) >= n_stories:
                break

        logger.info(f"UG generation complete — role={role}, produced {len(all_stories)} stories")
        return all_stories[:n_stories]

    def _extract_proposer_offer(self, text: str, total_amount: int) -> Optional[str]:
        m = re.search(r'<decision>\s*\$?([0-9]+)\s*</decision>', text, re.IGNORECASE)
        if m:
            try:
                offer = int(m.group(1))
                if 0 <= offer <= total_amount:
                    logger.debug(f"Extracted proposer offer: {offer}")
                    return str(offer)
                logger.warning(f"Proposer offer {offer} out of range 0..{total_amount}")
            except ValueError:
                logger.warning(f"Invalid proposer offer format: {m.group(1)}")
        else:
            logger.debug("No proposer <decision> tag found")
        return None

    def _extract_ultimatum(self, text: str, total_amount: int) -> Tuple[Optional[int], Optional[str]]:
        offer    = self._extract_offer(text)
        response = self._extract_response(text)

        if offer is not None:
            try:
                offer_int = int(offer)
            except Exception:
                logger.warning(f"Invalid offer format: {offer}")
                return None, response
            if not (0 <= offer_int <= total_amount):
                logger.warning(f"Offer {offer_int} out of range 0..{total_amount}")
                return None, response
            offer = offer_int

        if response:
            response_norm = response.strip().lower()
            if response_norm not in {"accept", "reject"}:
                logger.warning(f"Invalid response value: {response}")
                response = None
            else:
                response = response_norm

        return offer, response

    def _extract_offer(self, text: str) -> Optional[str]:
        m = re.search(r'<offer>\s*\$?([0-9]+)\s*</offer>', text, re.IGNORECASE)
        if m:
            logger.debug(f"Extracted offer: {m.group(1)}")
            return m.group(1)
        logger.debug("No <offer> tag found")
        return None

    def _extract_response(self, text: str) -> Optional[str]:
        m = re.search(r'<response>\s*(accept|reject)\s*</response>', text, re.IGNORECASE)
        if m:
            logger.debug(f"Extracted response: {m.group(1)}")
            return m.group(1)
        m2 = re.search(r'<decision>\s*(accept|reject)\s*</decision>', text, re.IGNORECASE)
        if m2:
            logger.debug(f"Extracted decision as response: {m2.group(1)}")
            return m2.group(1)
        logger.debug("No <response> or <decision> found for UG")
        return None


class PGGStoryGenerator(StoryGenerator):

    def create_query_pgg(
        self,
        game: PublicGoodsGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> str:
        logger.debug("Creating PGG query prompt")
        endowment   = game.effective_endowment
        multiplier  = game.multiplier
        num_players = game.num_players
        stakes_instr = _stakes_instruction(game.stakes)

        prompt = f"""Write {number_of_stories} unique stories about {num_players} people deciding how much to contribute to a shared project or pool.

IMPORTANT: Each story MUST be wrapped in <story></story> tags.

Setup:
- {num_players} people each have: {stakes_instr}
  Express this as a concrete number in the story.
- Each person secretly decides how much to contribute (anywhere from $0 to ${endowment})
- All contributions go into a shared pool
- The pool is multiplied by {multiplier}
- The final amount is split equally among all {num_players} people
- Everyone keeps what they didn't contribute PLUS their equal share

Topic domain: {topic}
Relationship: {actor_type}

CRITICAL CASTING RULE: All {num_players} participants MUST be individual human beings.
Do NOT use corporations, governments, or countries. Choose an endowment amount that is
realistic and meaningful at an individual human scale for the scenario


{ACTOR_TYPES[actor_type]['description']}

Examples: {ACTOR_TYPES[actor_type]['types']}

{unique_prompt}

FORMAT REQUIREMENTS:
1. Start with <story>
2. Tell the story focusing on ONE person's decision
3. Don't mention "game" or show math
4. Make the other {num_players-1} people's contributions part of the scenario
5. End with these EXACT tags:

You are [NAME] in this scenario. You have ${endowment}. How much will you contribute (0 to {endowment})?
<justification>Your reasoning here</justification>
<contribution>number from 0 to {endowment}</contribution>
<decision>same number</decision>

6. Close with </story>

Example:
<story>
Four colleagues at TechStart were pooling money for a team development fund. Each had $20 to potentially contribute. The fund would be multiplied by 1.6 and split equally. Sarah considered her options carefully, knowing the others might contribute varying amounts.

You are Sarah in this scenario. You have $20. How much will you contribute (0 to 20)?
<justification>Contributing helps the team grow while keeping some personal flexibility.</justification>
<contribution>15</contribution>
<decision>15</decision>
</story>

Now write {number_of_stories} unique stories following this exact format."""

        return textwrap.dedent(prompt)

    async def generate_batch_pgg(
        self,
        game: PublicGoodsGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10,
    ) -> BatchGenerationResult:
        logger.info("Generating PGG batch of stories")
        prompt = self.create_query_pgg(
            game, topic, world_type, actor_type, unique_prompt, number_of_stories
        )

        try:
            resp = await self.api_client.generate(prompt, model="llama")
            content = resp["llama"] if isinstance(resp, dict) and "llama" in resp else resp
            if isinstance(content, dict):
                content = str(content)

            raw_stories = re.findall(r'<story>(.*?)</story>', content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories for PGG")

            if not raw_stories:
                logger.warning("No PGG stories found")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                contribution = self._extract_contribution(story_content, game.effective_endowment)
                decision_field = str(contribution) if contribution is not None else None

                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision_field,
                )
                stories.append(story)

            summaries = await asyncio.gather(
                *[self.generate_story_summary(s.content) for s in stories]
            )
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"PGG batch created {len(stories)} stories")
            return BatchGenerationResult(stories, summaries, new_unique_prompt)

        except Exception as e:
            logger.error(f"Error generating PGG batch: {e}")
            return BatchGenerationResult([], [], "")

    async def generate_stories_pgg(
        self,
        game: PublicGoodsGame,
        topic: str,
        world_type: str,
        actor_type: str,
        n_stories: int = 100,
        batch_size: int = 10,
        unique_prompt: str = "",
    ) -> List[Story]:
        logger.info(f"Starting PGG generation of {n_stories} stories")

        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories: List[Story] = []
        n_batches = (n_stories + batch_size - 1) // batch_size
        number_of_stories = n_stories if n_stories < 10 else 10

        for batch_num in range(n_batches):
            logger.info(f"PGG generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch_pgg(
                game, topic, world_type, actor_type, unique_prompt, number_of_stories
            )

            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                unique_prompt = batch_result.unique_prompt
            else:
                logger.warning(f"PGG batch {batch_num + 1} produced no stories")

            if len(all_stories) >= n_stories:
                break

        logger.info(f"PGG generation complete. Produced {len(all_stories)} stories")
        return all_stories[:n_stories]

    def _extract_contribution(self, text: str, max_contribution: int) -> Optional[int]:
        m = re.search(r'<contribution>\s*([0-9]+)\s*</contribution>', text, re.IGNORECASE)
        if m:
            try:
                contrib = int(m.group(1))
                if 0 <= contrib <= max_contribution:
                    logger.debug(f"Extracted contribution: {contrib}")
                    return contrib
                logger.warning(f"Contribution {contrib} out of range 0..{max_contribution}")
                return None
            except ValueError:
                logger.warning(f"Invalid contribution format: {m.group(1)}")
                return None

        m2 = re.search(r'<decision>\s*([0-9]+)\s*</decision>', text, re.IGNORECASE)
        if m2:
            try:
                contrib = int(m2.group(1))
                if 0 <= contrib <= max_contribution:
                    logger.debug(f"Extracted contribution from decision tag: {contrib}")
                    return contrib
                logger.warning(f"Decision {contrib} out of range 0..{max_contribution}")
                return None
            except ValueError:
                logger.warning(f"Invalid decision format: {m2.group(1)}")
                return None

        logger.debug("No <contribution> or <decision> found for PGG")
        return None