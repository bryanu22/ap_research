# example_pgg.py
import asyncio
from game_theory_llm import PublicGoodsGame, PGGStoryGenerator, APIClient

async def main():
    game = PublicGoodsGame(
        num_players=4,
        endowment=20,
        multiplier=1.6
    )

    api_client = APIClient()
    
    generator = PGGStoryGenerator(api_client)
    
    stories = await generator.generate_stories_pgg(
        game=game,
        topic="business",
        world_type="real world",
        actor_type="allies",
        n_stories=1
    )
    
    for i, story in enumerate(stories, 1):
        print(f"\nStory {i}:")
        print(story.content)
        print(f"\nDecision (contribution): {story.decision}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())