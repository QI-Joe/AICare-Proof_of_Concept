import pyaudio
import asyncio
import websockets
import numpy as np
import time
import base64
import json


CHUNK_DURATION_MS = 100
RATE = 16000
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) 
SILENCE_THRESHOLD = 500  # Adjust this threshold based on your environment
SILENCE_DURATION = 2.0  # seconds
SEND_INTERVAL = 5.0  # seconds - accumulate audio for this duration before sending

async def real_speech():
    """
    function: use computer microphone to capture real speech and send to server for testing
    technique challenge: when to decide the end of speech?
    1. fixed time windows
    2. voice activity detection (VAD)
    3. user manual control (press a key to start/stop)
    
    stream transmission while not be considered to implement now.
    """
    uri = "ws://localhost:8765"
    p = pyaudio.PyAudio()
    
    # Find available input device
    input_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_device_index = i
            print(f"Using input device {i}: {device_info['name']}")
            break
    
    if input_device_index is None:
        print("No input device found. This might be a WSL issue.")
        p.terminate()
        return
    
    # this is writer of client side sending
    loop = asyncio.get_event_loop()
    stream = p.open(format=pyaudio.paInt16,
                    channels = 1,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK_SIZE,) # 底层缓冲池，声卡驱动存录了多少数据
    
    # Time tracking variables
    last_sound_time = time.time()
    is_speaking = False
    temporally_time = time.time()
    
    # Audio accumulation variables
    accumulated_audio = []  # List to store audio chunks
    accumulation_start_time = time.time()  # Track when accumulation started
    total_accumulated_duration = 0.0  # Track accumulated duration in seconds
    
    async with websockets.connect(uri) as websocket:
        print("Connected to server. Start speaking...")
        
        while True:
            try:
                # stream.read CHUNK_SIZE, 有多少数据要被取走，是跟着底层缓冲池来的，缓冲池满了才会往外取数据，所以最好两者设置一样, 至少 frames_per_buffer >= stream.read CHUNK_SIZE
                data = await loop.run_in_executor(None, stream.read,
                                                  CHUNK_SIZE, False) 
                audio_np = np.frombuffer(data, dtype=np.int16)
                volume = np.linalg.norm(audio_np) / np.sqrt(len(audio_np))
                
                audio_np = (audio_np).astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
                
                if volume > SILENCE_THRESHOLD:
                    last_sound_time = time.time()
                    is_speaking = True
                
                else:
                    # 在状态标记为讲话但长时间无声音，则认为讲话结束
                    if is_speaking and (time.time() - last_sound_time) > SILENCE_DURATION:
                        print("Silence detected, stopping accumulation.")
                        is_speaking = False
                
                # Accumulate audio chunks when speaking
                if is_speaking or volume > SILENCE_THRESHOLD:
                    accumulated_audio.append(audio_np)
                    chunk_duration = len(audio_np) / RATE  # Duration of this chunk in seconds
                    total_accumulated_duration += chunk_duration
                    
                    current_time = time.time()
                    elapsed_since_start = current_time - accumulation_start_time
                    
                    # Check if it's time to send (every SEND_INTERVAL seconds)
                    if elapsed_since_start >= SEND_INTERVAL:
                        # Concatenate all accumulated chunks
                        combined_audio = np.concatenate(accumulated_audio)
                        
                        print(f"Sending {total_accumulated_duration:.2f}s of audio ({len(accumulated_audio)} chunks)")
                        
                        encoded_audio = base64.b64encode(combined_audio.tobytes()).decode('utf-8')
                        message = json.dumps({
                            "action": "asr",
                            "audio": encoded_audio,
                            "sample_rate": 16_000,
                            "id": f"speech_{int(current_time)}",
                            "timestamp": current_time,
                            "data_type": "np.float32",
                            "duration": total_accumulated_duration,
                        })
                        await websocket.send(message)
                        
                        # Wait for response
                        result = await websocket.recv()
                        result = json.loads(result)
                        print(f"Received from server: {result}")
                        
                        # Reset accumulation
                        accumulated_audio = []
                        accumulation_start_time = time.time()
                        total_accumulated_duration = 0.0
                
                if time.time() - temporally_time > 600:
                    print("Ending session after 5 minutes.")
                    break
                
            except Exception as e:
                print(f"Error: {e}")
                break
            finally:
                await asyncio.sleep(0.01)  # Slight delay to prevent CPU overload
        stream.stop_stream()
        stream.close()
        p.terminate()
        
async def main():
    await real_speech()
    
    
def audio_test():
    p = pyaudio.PyAudio()
    
    # Find available input device
    input_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_device_index = i
            print(f"Using input device {i}: {device_info['name']}")
            break
    
    if input_device_index is None:
        print("No input device found. This might be a WSL issue.")
        p.terminate()
        return
    
if __name__ == "__main__":
    # audio_test()
    asyncio.run(main())
    