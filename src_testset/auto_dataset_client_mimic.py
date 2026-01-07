import asyncio
from datasets import load_dataset
import numpy as np
from src_backend.RPC_client import RPClient
import time

test_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
text_dataset = ...

async def single_request_test(client: RPClient, sample):
    """Test single request"""
    print("=== Single Request Test ===")
    
    audio_data = sample['audio']
    audio_array = audio_data['array'].astype(np.float32)
    audio_sample_rate = audio_data['sampling_rate']
    
    print(f"Sending: {sample['id']} (sample_rate: {audio_sample_rate} Hz)")
    
    result = await client.call_asr(audio_array, audio_sample_rate)
    
    print(f"Received: {result}")
    print(f"Ground truth: {sample['text']}\n")

async def concurrent_requests_test(client: RPClient, data, num_requests: int = 10):
    """Test concurrent requests"""
    print(f"=== Concurrent Requests Test ({num_requests} samples) ===")
    
    async def send_request(sample):
        try:
            client_id = sample['id']
            audio_data = sample['audio']
            audio_array = audio_data['array'].astype(np.float32)
            audio_sample_rate = audio_data['sampling_rate']
            
            start_time = time.time()
            result = await client.call_asr(audio_array, audio_sample_rate)
            elapsed = time.time() - start_time
            
            print(f"[Client {client_id}] Response received (elapsed: {elapsed:.2f}s): {result['text']}")
            return result
            
        except Exception as e:
            print(f"[Client {client_id}] Error: {e}")
            return None
    
    num_requests = min(num_requests, len(data))
    tasks = [send_request(data[i]) for i in range(num_requests)]
    
    print("Ground truths:")
    for i in range(min(10, num_requests)):
        print(f"  {i}: {data[i]['text']}")
    
    results = await asyncio.gather(*tasks)
    print(f"\n{num_requests} tasks completed")
    return results

async def run_batch_test_for_api(num_samples: int = 10):
    """
    API-compatible function for frontend_api.py
    Returns results list without printing
    """
    client = RPClient()
    await client.connect()
    
    try:
        results = []
        num_requests = min(num_samples, len(test_dataset))
        
        for i in range(num_requests):
            sample = test_dataset[i]
            audio_data = sample['audio']
            audio_array = audio_data['array'].astype(np.float32)
            audio_sample_rate = audio_data['sampling_rate']
            
            try:
                result = await client.call_asr(audio_array, audio_sample_rate)
                results.append({
                    "id": sample['id'],
                    "recognized": result.get("text", ""),
                    "ground_truth": sample['text']
                })
            except Exception as e:
                results.append({
                    "id": sample['id'],
                    "recognized": f"Error: {str(e)}",
                    "ground_truth": sample['text']
                })
        
        return results
    finally:
        await client.close()

async def main():
    """Main test function"""
    print("=" * 60)
    print("RabbitMQ RPC ASR System Test")
    print("=" * 60)
    print()
    
    client = RPClient()
    await client.connect()
    
    try:
        # Test 1: Single request
        sample = test_dataset[0]
        await single_request_test(client, sample)
        await asyncio.sleep(1)
        
        # Test 2: Concurrent requests
        await concurrent_requests_test(client, test_dataset, num_requests=10)
        await asyncio.sleep(1)
    finally:
        await client.close()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")