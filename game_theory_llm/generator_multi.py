# game_theory_llm/generator.py
from typing import List, Tuple, Optional
import re
import logging
from .models import PayoffMatrix, Story
from .api import APIClient
from .config import WORLD_DICT, ACTOR_TYPES, TOPICS
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from .models import PayoffMatrix, Story, UltimatumGame

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchGenerationResult:
    """Results from a batch of story generation."""
    stories: List[Story]
    summaries: List[str]
    unique_prompt: str

class StoryGenerator:
    """Generates stories from game theory scenarios."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("StoryGenerator initialized")

    async def generate_story_summary(self, story: str) -> str:
        """Generate a concise, structured summary of a story using the API."""
        logger.debug("Generating summary for story")
        prompt = """Analyze this story and create a single, comprehensive sentence that captures:
        1. The main character(s) and their defining traits
        2. The primary setting/location
        3. The core conflict or goal
        4. The most important plot development

        Story to summarize:
        {story}"""

        try:
            summary = await self.api_client.generate(prompt.format(story=story))
            logger.debug(f"Generated summary: {summary[:100]}...")
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"A story about {story[:100]}..."

    def generate_unique_prompt(self, summaries: List[str]) -> str:
        """Generate a prompt that helps ensure unique stories based on previous summaries."""
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
        unique_prompt: str = ""
    ) -> str:
        """Create a prompt using the original format."""
        logger.debug("Creating query prompt")
        prompt = f"""Write {10} unique stories about a scenario involving two agents and their possible actions:

Matrix: {matrix.format_matrix()}

Topic: {topic}
Relationship: {world_type} {actor_type}

Each agent has two decisions (A or B):
- Both choose A: {matrix.matrix[0]}
- Agent 1 A, Agent 2 B: {matrix.matrix[1]}
- Agent 1 B, Agent 2 A: {matrix.matrix[2]}
- Both choose B: {matrix.matrix[3]}

{WORLD_DICT[world_type]}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}


Rules:
1. Do not under any cirumstance mention that this is a game
2. Label decisions as A or B
3. Don't show explicit payoffs
4. Show how outcomes depend on both agents
5. Surround each story with <story></story> tags

{unique_prompt}

End each story with: 
<toadd>
The name of agent 1 is <name> <agent1> </name> 
The name of agent 1 is <name> <agent2> </name> 
<justification></justification>
<decision>A or B</decision>
</toadd>"""
        
        logger.debug(f"Created prompt of length {len(prompt)}")
        return prompt

    async def generate_batch(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = ""
    ) -> BatchGenerationResult:
        """Generate a batch of stories with summaries."""
        logger.info("Generating batch of stories")
        prompt = self.create_query(payoff_matrix, topic, world_type, actor_type, unique_prompt)
        
        try:
            content = await self.api_client.generate(prompt)
            content = content["llama"]
            logger.debug(f"Generated content length: {len(content)}")
            
            # Extract stories
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
                    decision=decision
                )
                stories.append(story)
            
            # Generate summaries
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
        batch_size: int = 10
    ) -> List[Story]:
        """Generate multiple stories in batches."""
        logger.info(f"Starting generation of {n_stories} stories in batches of {batch_size}")
        
        # Input validation
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
        
        for batch_num in range(n_batches):
            logger.info(f"Generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch(
                payoff_matrix, 
                topic, 
                world_type, 
                actor_type, 
                unique_prompt
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
        """Extract decision from story."""
        decision_pattern = r'<decision>\s*([AB])\s*</decision>'
        match = re.search(decision_pattern, text)
        if match:
            logger.debug(f"Extracted decision: {match.group(1)}")
            return match.group(1)
        logger.warning("No decision found in story")
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
        """
        Create a prompt for Ultimatum Game stories.
        Each story must include:
         - <story> ... </story>
         - <offer>NUM</offer>    (proposer's offer to responder)
         - <response>accept|reject</response>
         - <justification>...</justification>
         - <decision>A|B</decision>  (map to A=accept, B=reject for compatibility)
        The returned Story.decision will be set to "offer:response" (e.g. "40:accept").
        """
        logger.debug("Creating ultimatum query prompt")
        total = game.total_amount
        prompt = f"""
Write {number_of_stories} unique short stories (word-problem style) involving two agents:
one makes a monetary offer (the proposer) and the other accepts or rejects (the responder).
Do not call this a game. Keep it natural and varied.

Context:
- Total amount to split: {total}.
- The proposer will choose an integer offer between 0 and {total} inclusive.
- The responder will either accept or reject the offer.
- If the responder accepts, the proposer keeps (total - offer) and the responder gets (offer).
- If the responder rejects, both get 0.

Topic: {topic}
Relationship: {world_type} {actor_type}

{WORLD_DICT.get(world_type, "")}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

Rules for generation:
1. Surround each story with <story>...</story> tags.
2. At the end of each story include these tags (replace placeholders):
   <offer>number</offer>
   <response>accept</response> or <response>reject</response>
   <justification>...</justification>
   <decision>A</decision> for accept or <decision>B</decision> for reject
3. Do not print explicit payoff arithmetic; show consequences in narrative.
4. Vary names, settings, and motives. Avoid reusing elements from the provided unique prompt.

{unique_prompt}

Example ending to imitate (but vary wording):
<offer>30</offer>
<response>accept</response>
<justification>The responder needs the funds to pay for...</justification>
<decision>A</decision>
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
        """Generate a batch of Ultimatum stories with extracted offers/responses."""
        logger.info("Generating UG batch of stories")
        prompt = self.create_query_ultimatum(game, topic, world_type, actor_type, unique_prompt, number_of_stories)

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

            stories: List[Story] = []
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
        """Generate multiple UG stories in batches."""
        logger.info(f"Starting UG generation of {n_stories} stories")
        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")

        all_stories: List[Story] = []
        unique_prompt = ""
        n_batches = (n_stories + batch_size - 1) // batch_size

        for batch_num in range(n_batches):
            logger.info(f"UG generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch_ultimatum(
                game,
                topic,
                world_type,
                actor_type,
                unique_prompt,
                min(batch_size, n_stories - len(all_stories))
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


    def _extract_ultimatum(self, text: str, total_amount: int) -> (Optional[int], Optional[str]):
        """
        Extract <offer> and <response> from a story.
        Returns (offer:int or None, response:'accept'|'reject' or None)
        """
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
        """Find numeric offer inside <offer> tags."""
        m = re.search(r'<offer>\s*([0-9]+)\s*</offer>', text, re.IGNORECASE)
        if m:
            logger.debug(f"Extracted offer: {m.group(1)}")
            return m.group(1)
        logger.debug("No <offer> tag found")
        return None

    def _extract_response(self, text: str) -> Optional[str]:
        """Find response inside <response> tags (accept/reject)."""
        m = re.search(r'<response>\s*(accept|reject)\s*</response>', text, re.IGNORECASE)
        if m:
            logger.debug(f"Extracted response: {m.group(1)}")
            return m.group(1)
        m2 = re.search(r'<decision>\s*([ABab])\s*</decision>', text)
        if m2:
            d = m2.group(1).upper()
            mapped = "accept" if d == "A" else "reject"
            logger.debug(f"Extracted decision tag maps to response: {d} -> {mapped}")
            return mapped
        logger.debug("No <response> or <decision> found for UG")
        return None
