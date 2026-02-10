# game_theory_llm/generator.py
from typing import List, Tuple, Optional
import re
import logging
from .models import PayoffMatrix, Story, UltimatumGame, PublicGoodsGame
from .api import APIClient
from .config import WORLD_DICT, ACTOR_TYPES, TOPICS
import asyncio
from dataclasses import dataclass
import textwrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            summary =  summary["llama"]
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
        number_of_stories: int = 10
    ) -> str:
        logger.debug("Creating query prompt")
        prompt = f"""Write {number_of_stories} unique stories about a scenario involving two agents and their possible actions.
        This matrix {matrix.format_matrix()} represents each agent's happiness based on their decision and the other agent's decision.
        The topic you need to write about is {topic}.
        The relationship between the two agents is {world_type} {actor_type}.

        Please write {number_of_stories} stories that would present this situation as a word problem having to do with {topic} without making it obvious that this is based on a game theory problem. Be creative and varied in your story structures and motifs.The relationship between the agents should be that they are {world_type} {actor_type}.

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
        1. Do not under any cirumstance mention that this is a game
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
        
        prompt = textwrap.dedent(prompt)
        logger.debug(f"Created prompt of length {len(prompt)}")
        return prompt

    async def generate_batch(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10
    ) -> BatchGenerationResult:
        logger.info("Generating batch of stories")
        prompt = self.create_query(payoff_matrix, topic, world_type, actor_type, unique_prompt, number_of_stories)
        
        try:
            content = await self.api_client.generate(prompt, model="llama")
            content = content["llama"]
            logger.debug(f"Generated content length: {len(content)}")
            
            story_pattern = r'<story>(.*?)</story>'
            raw_stories = re.findall(story_pattern, content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories from response")
            
            if not raw_stories:
                logger.warning("No stories found in generated content")
                logger.debug(f"Content preview: {content[:500]}...")
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
                    decision=decision
                )
                stories.append(story)
            
            summaries = await asyncio.gather(
                *[self.generate_story_summary(story.content) for story in stories]
            )
            print(f"Summaries: {summaries}")
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
        batch_size: int = 10
    ) -> List[Story]:
        logger.info(f"Starting generation of {n_stories} stories in batches of {batch_size}")
        
        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")
        
        all_stories = []
        all_summaries = []
        unique_prompt = ""
        
        n_batches = (n_stories + batch_size - 1) // batch_size
        logger.info(f"Will generate {n_batches} batches")
        if n_stories < 10:
            number_of_stories = n_stories
        else:
            number_of_stories = 10
        for batch_num in range(n_batches):
            logger.info(f"Generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch(
                payoff_matrix, 
                topic, 
                world_type, 
                actor_type, 
                unique_prompt,
                number_of_stories
            )
            
            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                all_summaries.extend(batch_result.summaries)
                unique_prompt = batch_result.unique_prompt
                logger.info(f"Added {len(batch_result.stories)} stories from batch {batch_num + 1}")
            else:
                logger.warning(f"Batch {batch_num + 1} generated no stories")
            
            if len(all_stories) >= n_stories:
                logger.info(f"Reached target number of stories ({n_stories})")
                break
        
        logger.info(f"Generation complete. Generated {len(all_stories)} stories total")
        return all_stories[:n_stories]

    def _extract_decision(self, text: str) -> Optional[str]:
        patterns = [
            (r'<decision>\s*([AB])\s*</decision>', {'A': 'A', 'B': 'B'}),
            (r'<decision>\s*([12])\s*</decision>', {'1': 'A', '2': 'B'}),
            (r'<decision>\s*(\&{1,2})\s*</decision>', {'&': '&', '&&': '&&'}),
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


class UltimatumStoryGenerator(StoryGenerator):
    
    def create_query_ultimatum(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10
    ) -> str:
        logger.debug("Creating ultimatum query prompt")
        total = game.total_amount
        
        prompt = f"""Write {number_of_stories} unique short stories involving two agents:
one makes a monetary offer (the proposer) and the other accepts or rejects (the responder).
Do not call this a game. Keep it natural and varied.

Context:
- Total amount to split: ${total}.
- The proposer will choose an integer offer between $0 and ${total} inclusive.
- The responder will either accept or reject the offer.
- If the responder accepts, the proposer keeps (total - offer) and the responder gets the offer amount.
- If the responder rejects, both get $0.

Topic: {topic}
Relationship: {world_type} {actor_type}

{WORLD_DICT.get(world_type, "")}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}

Rules for generation:
1. Surround each story with <story>...</story> tags.
2. At the end of each story, include these tags:
   <offer>number</offer>
   <response>accept</response> or <response>reject</response>
3. Do not print explicit payoff arithmetic; show consequences in narrative.
4. Vary names, settings, and motives. Avoid reusing elements from the provided unique prompt.
5. Make it clear which agent is the proposer and which is the responder.

{unique_prompt}

End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with the RESPONDER agent in the story:
<toadd>
You are <agent> in this scenario. The proposer has offered you <offer>$X</offer>. Do you accept or reject?
Provide your justification for your decision inside <justification></justification> tags.
Then, output your decision: <decision>accept</decision> or <decision>reject</decision>
</toadd>

Example story ending:
<offer>40</offer>
<response>accept</response>
<justification>The responder needs the funds and the split seems fair enough.</justification>
<decision>accept</decision>
"""
        prompt = textwrap.dedent(prompt)
        logger.debug(f"Created ultimatum prompt length {len(prompt)}")
        return prompt

    async def generate_batch_ultimatum(
        self,
        game: UltimatumGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10
    ) -> BatchGenerationResult:
        logger.info("Generating UG batch of stories")
        prompt = self.create_query_ultimatum(
            game, topic, world_type, actor_type, unique_prompt, number_of_stories
        )

        try:
            resp = await self.api_client.generate(prompt, model="llama")
            content = resp["llama"] if isinstance(resp, dict) and "llama" in resp else resp
            if isinstance(content, dict):
                content = str(content)
            logger.debug(f"UG generated content length: {len(content)}")
            
            story_pattern = r'<story>(.*?)</story>'
            raw_stories = re.findall(story_pattern, content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories for UG")
            
            if not raw_stories:
                logger.warning("No UG stories found")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                offer, response = self._extract_ultimatum(story_content, game.total_amount)
                decision_field = None
                if offer is not None and response is not None:
                    decision_field = f"{offer}:{response}"
                
                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision_field
                )
                stories.append(story)
            
            summaries = await asyncio.gather(
                *[self.generate_story_summary(s.content) for s in stories]
            )
            
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"UG batch created {len(stories)} stories")
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
        n_stories: int = 100,
        batch_size: int = 10
    ) -> List[Story]:
        logger.info(f"Starting UG generation of {n_stories} stories")
        
        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories = []
        unique_prompt = ""
        n_batches = (n_stories + batch_size - 1) // batch_size
        
        if n_stories < 10:
            number_of_stories = n_stories
        else:
            number_of_stories = 10

        for batch_num in range(n_batches):
            logger.info(f"UG generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch_ultimatum(
                game,
                topic,
                world_type,
                actor_type,
                unique_prompt,
                number_of_stories
            )
            
            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                unique_prompt = batch_result.unique_prompt
            else:
                logger.warning(f"UG batch {batch_num + 1} produced no stories")

            if len(all_stories) >= n_stories:
                break

        logger.info(f"UG generation complete. Produced {len(all_stories)} stories")
        return all_stories[:n_stories]

    def _extract_ultimatum(
        self, 
        text: str, 
        total_amount: int
    ) -> Tuple[Optional[int], Optional[str]]:
        offer = self._extract_offer(text)
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
        number_of_stories: int = 10
    ) -> str:
        logger.debug("Creating PGG query prompt")
        endowment = game.endowment
        multiplier = game.multiplier
        num_players = game.num_players
        
        prompt = f"""Write {number_of_stories} unique stories about {num_players} people deciding how much to contribute to a shared project or pool.

IMPORTANT: Each story MUST be wrapped in <story></story> tags.

Setup:
- {num_players} people each have ${endowment}
- Each person secretly decides how much to contribute (anywhere from $0 to ${endowment})
- All contributions go into a shared pool
- The pool is multiplied by {multiplier}
- The final amount is split equally among all {num_players} people
- Everyone keeps what they didn't contribute PLUS their equal share

Topic: {topic}
World: {world_type}
Relationship: {actor_type}

{WORLD_DICT.get(world_type, "")}

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
        
        prompt = textwrap.dedent(prompt)
        logger.debug(f"Created PGG prompt length {len(prompt)}")
        return prompt

    async def generate_batch_pgg(
        self,
        game: PublicGoodsGame,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = "",
        number_of_stories: int = 10
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
            logger.debug(f"PGG generated content length: {len(content)}")
            
            story_pattern = r'<story>(.*?)</story>'
            raw_stories = re.findall(story_pattern, content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories for PGG")
            
            if not raw_stories:
                logger.warning("No PGG stories found")
                return BatchGenerationResult([], [], "")

            stories = []
            for story_content in raw_stories:
                contribution = self._extract_contribution(story_content, game.endowment)
                decision_field = str(contribution) if contribution is not None else None
                
                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    prompt=prompt,
                    decision=decision_field
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
        batch_size: int = 10
    ) -> List[Story]:
        logger.info(f"Starting PGG generation of {n_stories} stories")
        
        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories = []
        unique_prompt = ""
        n_batches = (n_stories + batch_size - 1) // batch_size
        
        if n_stories < 10:
            number_of_stories = n_stories
        else:
            number_of_stories = 10

        for batch_num in range(n_batches):
            logger.info(f"PGG generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch_pgg(
                game,
                topic,
                world_type,
                actor_type,
                unique_prompt,
                number_of_stories
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
                else:
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
                else:
                    logger.warning(f"Decision {contrib} out of range 0..{max_contribution}")
                    return None
            except ValueError:
                logger.warning(f"Invalid decision format: {m2.group(1)}")
                return None
        
        logger.debug("No <contribution> or <decision> found for PGG")
        return None
