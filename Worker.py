import aio_pika
import json
from typing import List, Optional, Dict, Any, Callable
from ASR_model import ASRModel
import base64
import numpy as np
from log import get_logger
import asyncio

logger = get_logger()


class Worker:
    def __init__(self):
        self.asr_model = ASRModel("openai-whisper-small")
        self.llm_model: Optional[Any] = None
        self.connection: Optional[Any | aio_pika.Connection] = None
        self.channel: Optional[Any | aio_pika.Channel] = None
        # self.asyncio_key = asyncio.get_running_loop()
        
        # Check and log model status
        status = self.asr_model.check_model_status()
        logger.info("=" * 60)
        logger.info("ASR Model Status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60, "\n")
    
    async def initial_connection(self):
        self.connection = await aio_pika.connect_robust(
            host='localhost', 
            login='guest', password='guest',
            heartbeat = 60, 
        )
        
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=2)
        
        asr_queue = await self.channel.declare_queue(
            name = 'asr_queue', durable=True
        )
        llm_queue = await self.channel.declare_queue(
            name='llm_queue', durable=True
        )
        self.queue_dict: Dict[str, aio_pika.Queue] = {
            'asr': asr_queue,
            'llm': llm_queue    
        }
        
        for key, queued in self.queue_dict.items():
            purged_count = await queued.purge()
            logger.info(f"queue {key} purged out {purged_count} old messages")
            await queued.bind(self.channel.default_exchange, routing_key=key)
    
    def asr_message_process(self, data:Dict):
        audio_data = np.frombuffer(
            base64.b64decode(data["audio"]),
            dtype=np.float32
        )
        sample_rate = data['sample_rate']
        recognized_text = self.asr_model.process(audio_data, sample_rate)
        
        result = {"text": recognized_text}
        return result
    
    def llm_message_process(self, data: Dict):
        
        result = {"text": "LLM demo return"}
        return result
    
    async def on_message(self, process_fn: Callable, message: aio_pika.IncomingMessage):
        async with message.process():  # Auto-acknowledges on success
            try:
                task_info = json.loads(message.body)
                
                result = process_fn(task_info)
                
                # 4. Send response back to client
                await self.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(result).encode(),
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )
                
                logger.info(f"✓ Processed {message.correlation_id[:8]}")
                
            except Exception as e:
                logger.error(f"✗ Error processing message: {e}")
                
    async def run(self):
        if self.channel is None or self.connection is None:
            raise RuntimeError("Connection not initialized. Call initial_connection() first!")
        
        logger.info("Starting message consumers...")
        
        # Map queue names to their processing functions
        process_fn_map = {
            'asr': self.asr_message_process,
            'llm': self.llm_message_process,
        }
        
        # Elegant loop: consume from all queues
        for queue_name, process_fn in process_fn_map.items():
            queue = self.queue_dict[queue_name]
            
            # Create callback that binds the specific process function
            async def create_callback(fn):
                async def callback(message: aio_pika.IncomingMessage):
                    await self.on_message(fn, message)
                return callback
            
            # Start consuming
            await queue.consume(await create_callback(process_fn))
            logger.info(f"✅ {queue_name.upper()} queue consuming")
        
        logger.info("✅ All queues ready, waiting for messages...")
        
        # Keep running forever
        await asyncio.Future()
        
        
async def main():
    worker = Worker()
    try:
        await worker.initial_connection()
        await worker.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Worker stopped by user")
        print("\n⚠️  Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        print(f"❌ Worker error: {e}")
    finally:
        if worker.connection and not worker.connection.is_closed:
            await worker.connection.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    asyncio.run(main())
        
        