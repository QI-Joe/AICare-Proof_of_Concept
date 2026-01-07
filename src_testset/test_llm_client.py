import asyncio
from src_backend.RPC_client import RPClient

async def test_llm():
    """Test LLM functionality"""
    print("=" * 60)
    print("Testing LLM RPC Call")
    print("=" * 60)
    
    client = RPClient()
    await client.connect()
    
    try:
        # Test LLM with a simple prompt
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me a short joke.",
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[Test {i}] Sending prompt: {prompt}")
            result = await client.call_llm(prompt, timeout=60.0)
            print(f"[Response] {result['text']}")
            print("-" * 60)
            await asyncio.sleep(1)
        
    finally:
        await client.close()
    
    print("\n" + "=" * 60)
    print("LLM Test Completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_llm())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
