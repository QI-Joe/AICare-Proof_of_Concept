#!/usr/bin/env python3
# test_client.py - 测试 WebSocket + RabbitMQ 系统的客户端

import asyncio
import websockets
import json
import time
from datasets import load_dataset
import base64
import pyaudio
import numpy as np

test_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
CHUNK_SIZE = 1

async def single_request_test(sample):
    """测试单个请求"""
    print("=== 单请求测试 ===")
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            # 获取测试数据
            audio_data = sample['audio']
            audio_array, audio_sample_rate = audio_data['array'].astype(np.float32), audio_data['sampling_rate']
            print(audio_sample_rate, "audio sampling rate")
            
            # 发送测试音频数据
            test_data = json.dumps({
                "action": "asr",
                "audio": base64.b64encode(audio_array.tobytes()).decode('utf-8'),
                "sample_rate": audio_sample_rate,
                "id": sample['id'],
                "timestamp": time.time(),
                "data_type": "np.float32",
            })
            
            print(f"发送: {sample['id']} (sample_rate: {audio_sample_rate} Hz)")
            await websocket.send(test_data)
            
            # 等待响应
            response = await websocket.recv()
            response = json.loads(response)
            print(f"收到: {response}")
            print(f"Ground truth: {sample['text']}\n")
            
    except Exception as e:
        print(f"错误: {e}\n")

async def concurrent_requests_test(data):
    """测试并发多请求"""
    print("=== 并发请求测试 ===")
    uri = "ws://localhost:8765"
    
    async def send_request(sample):
        try:
            async with websockets.connect(uri) as websocket:
                # Extract data from the sample WITHOUT modifying original
                client_id: str = sample['id']
                raw_audio = sample['audio']
                audio_array, audio_sample_rate = raw_audio['array'].astype(np.float32), raw_audio['sampling_rate']
                
                # Create a NEW clean dictionary for JSON serialization
                request_data = {
                    "action": "asr",
                    "audio": base64.b64encode(audio_array.tobytes()).decode('utf-8'),
                    "sample_rate": audio_sample_rate,
                    "id": client_id,
                    "timestamp": time.time(),
                    "data_type": "np.float32",
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(request_data))
                response = await websocket.recv()
                response = json.loads(response)
                
                elapsed = time.time() - start_time
                print(f"[客户端{client_id}] 收到响应 (耗时: {elapsed:.2f}秒): {response}")
                
        except Exception as e:
            print(f"[客户端{client_id}] 错误: {e}")
    
    # Send multiple concurrent requests (changed from 1 to 3 for actual testing)
    num_requests = min(100, len(data))  # Limit to 3 concurrent or data length
    tasks = [send_request(data[i]) for i in range(num_requests)]
    
    for i in range(num_requests if num_requests < 10 else 10):
        print(i, data[i]['text'])
    await asyncio.gather(*tasks)
    print(f"{num_requests} tasks sent and completed")


async def main():
    """主测试函数"""
    print("=" * 60)
    print("WebSocket + RabbitMQ ASR 系统测试")
    print("=" * 60)
    print()
    
    # 测试1: 单个请求
    sample = test_dataset[0]
    await single_request_test(sample=sample)
    await asyncio.sleep(1)
    
    # 测试2: 并发请求
    await concurrent_requests_test(test_dataset)
    await asyncio.sleep(1)
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)

# === 新增: 供 API 调用的封装函数 ===
async def run_batch_test_for_api(num_samples: int = 10):
    """
    供 frontend_api.py 调用的批量测试函数
    返回结果列表，不打印到控制台
    """
    uri = "ws://localhost:8765"
    results = []
    
    async def send_request(sample):
        try:
            async with websockets.connect(uri) as websocket:
                client_id = sample['id']
                audio_data = sample['audio']
                audio_array, audio_sample_rate = audio_data['array'].astype(np.float32), audio_data['sampling_rate']
                
                request_data = {
                    "action": "asr",
                    "audio": base64.b64encode(audio_array.tobytes()).decode('utf-8'),
                    "sample_rate": audio_sample_rate,
                    "id": client_id,
                    "timestamp": time.time(),
                    "data_type": "np.float32",
                }
                
                await websocket.send(json.dumps(request_data))
                response = await websocket.recv()
                response = json.loads(response)
                
                return {
                    "id": client_id,
                    "recognized": response.get("text", ""),
                    "ground_truth": sample['text']
                }
        except Exception as e:
            return {
                "id": client_id,
                "recognized": f"错误: {str(e)}",
                "ground_truth": sample['text']
            }
    
    num_requests = min(num_samples, len(test_dataset))
    tasks = [send_request(test_dataset[i]) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    return results

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试失败: {e}")

