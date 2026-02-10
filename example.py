# example.py
import asyncio
from game_theory_llm import UltimatumGame, UltimatumStoryGenerator, APIClient

async def main():
    game = UltimatumGame(total_amount=100)

    api_client = APIClient()
    
    generator = UltimatumStoryGenerator(api_client)
    
    stories = await generator.generate_stories_ultimatum(
        game=game,
        topic="business",
        world_type="real world",
        actor_type="allies",
        n_stories=2
    )
    
    for i, story in enumerate(stories, 1):
        print(f"\nStory {i}:")
        print(story.content)
        print(f"\nDecision: {story.decision}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())