import asyncio
from game_theory_llm import UltimatumGame, UltimatumStoryGenerator, APIClient, Stakes

async def main():
    game = UltimatumGame(total_amount=20, stakes=Stakes.LOW)

    api_client = APIClient()
    generator = UltimatumStoryGenerator(api_client)

    proposer_stories = await generator.generate_stories_ultimatum(
        game=game,
        topic="business",
        world_type="real world",
        actor_type="allies",
        role="proposer",
        n_stories=2,
    )

    responder_stories = await generator.generate_stories_ultimatum(
        game=game,
        topic="business",
        world_type="real world",
        actor_type="allies",
        role="responder",
        n_stories=2,
    )

    print("=" * 80)
    print("PROPOSER STORIES")
    print("=" * 80)
    for i, story in enumerate(proposer_stories, 1):
        print(f"\nStory {i}:")
        print(story.content)
        print(f"\nOffer made: ${story.decision}")
        print("-" * 80)

    print("=" * 80)
    print("RESPONDER STORIES")
    print("=" * 80)
    for i, story in enumerate(responder_stories, 1):
        print(f"\nStory {i}:")
        print(story.content)
        print(f"\nDecision: {story.decision}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())