import asyncio
import aio_pika
import logging
from datetime import datetime
from RPC_client import RPClient
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StabilityTest")

async def connection_heartbeat_test(duration_minutes: int = 480):
    """Test connection stays alive with heartbeat"""
    logger.info(f"Starting {duration_minutes}-minute connection stability test")
    
    connection = await aio_pika.connect_robust(
        "amqp://guest:guest@localhost/?heartbeat=60",
    )
    
    logger.info(f"[{datetime.now()}] Connection established")
    
    async with connection:
        end_time = asyncio.get_event_loop().time() + (duration_minutes * 60)
        iteration = 0
        
        while asyncio.get_event_loop().time() < end_time:
            await asyncio.sleep(50)
            iteration += 1
            
            if connection.is_closed:
                logger.error(f"[{datetime.now()}] Connection LOST at iteration {iteration}!")
                break
            else:
                logger.info(f"[{datetime.now()}] Iteration {iteration}: Connection healthy ✓")
    
    logger.info("Test completed")

async def request_stress_test(num_requests: int = 100, concurrent: int = 10):
    """Test system under sustained load"""
    logger.info(f"Starting stress test: {num_requests} requests, {concurrent} concurrent")
    
    client = RPClient()
    await client.connect()
    
    dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    
    async def single_request(request_id: int):
        try:
            start = asyncio.get_event_loop().time()
            result = await client.call_asr(dummy_audio, 16000)
            duration = asyncio.get_event_loop().time() - start
            logger.info(f"Request {request_id}: {duration:.2f}s - {result['text'][:30]}...")
            return True
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            return False
    
    successes = 0
    for batch_start in range(0, num_requests, concurrent):
        batch_end = min(batch_start + concurrent, num_requests)
        tasks = [single_request(i) for i in range(batch_start, batch_end)]
        results = await asyncio.gather(*tasks)
        successes += sum(results)
        
        logger.info(f"Progress: {batch_end}/{num_requests} ({successes} successful)")
        await asyncio.sleep(1)  # Brief pause between batches
    
    await client.close()
    
    logger.info(f"Test completed: {successes}/{num_requests} successful ({successes/num_requests*100:.1f}%)")

async def main():
    print("=" * 60)
    print("ASR System Stability Tests")
    print("=" * 60)
    print()
    print("Select test:")
    print("1. Connection heartbeat test (1 hour)")
    print("2. Request stress test (100 requests)")
    print("3. Both tests")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        await connection_heartbeat_test(duration_minutes=60)
    
    if choice in ['2', '3']:
        await request_stress_test(num_requests=100, concurrent=10)
    
    print("\n" + "=" * 60)
    print("All stability tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")