import aio_pika
import json
import sys
sys.path.append("../")
from typing import List, Optional, Dict, Any, Callable
from src_backend.ASR_model import ASRModel, LLModel
import base64
import numpy as np
from utils.log import get_logger
import asyncio

logger = get_logger()
asr_queue_name = "asr_queue"
llm_queue_name = "llm_queue"


class Worker:
    def __init__(self):
        # Initialize ASR model
        self.asr_model = ASRModel("openai-whisper-small")
        
        # Initialize LLM model
        logger.info("Loading LLM model...")
        self.llm_model = LLModel("qwen0.5b")
        
        self.connection: Optional[Any | aio_pika.Connection] = None
        self.channel: Optional[Any | aio_pika.Channel] = None
        # self.asyncio_key = asyncio.get_running_loop()
        
        # Check and log ASR model status
        asr_status = self.asr_model.check_model_status()
        logger.info("=" * 60)
        logger.info("ASR Model Status:")
        for key, value in asr_status.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60 + "\n")
        
        # Check and log LLM model status
        llm_status = self.llm_model.check_module_status()
        logger.info("=" * 60)
        logger.info("LLM Model Status:")
        for key, value in llm_status.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60 + "\n")
    
    async def initial_connection(self):
        self.connection = await aio_pika.connect_robust(
            host='localhost', 
            login='guest', password='guest',
            heartbeat = 60, 
        )
        
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=2)
        
        asr_queue = await self.channel.declare_queue(
            name = asr_queue_name, durable=True
        )
        llm_queue = await self.channel.declare_queue(
            name=llm_queue_name, durable=True
        )
        self.queue_dict: Dict[str, aio_pika.Queue] = {
            asr_queue_name: asr_queue,
            llm_queue_name: llm_queue    
        }
        
        for key, queued in self.queue_dict.items():
            purged_count = await queued.purge()
            logger.info(f"queue {key} purged out {purged_count} old messages")
            # No need to bind to default exchange - it auto-routes by queue name
    
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
        # Extract prompt from request
        prompt = data.get('text', '')
        
        if not prompt:
            logger.warning("Received empty prompt for LLM processing")
            return {"text": "Error: Empty prompt"}
        
        # Process with LLM model
        logger.info(f"Processing LLM request: {prompt[:50]}...")
        response_text = self.llm_model.process(prompt)
        
        result = {"text": response_text}
        logger.info(f"LLM response: {response_text[:100]}...")
        return result
    
    async def on_message(self, process_fn: Callable, queue_name: str, message: aio_pika.IncomingMessage):
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
                
                # Check remaining queue length after processing
                queue = self.queue_dict[queue_name]
                queue_info = await queue.declare(passive=True)
                remaining_count = queue_info.message_count
                
                if remaining_count > 0:
                    status_msg = f"📊 Queue '{queue_name}_queue' has {remaining_count} task(s) pending"
                    logger.info(status_msg)
                    print(status_msg)
                
            except Exception as e:
                logger.error(f"✗ Error processing message: {e}")
                
    async def run(self):
        if self.channel is None or self.connection is None:
            raise RuntimeError("Connection not initialized. Call initial_connection() first!")
        
        logger.info("Starting message consumers...")
        
        # Map queue names to their processing functions
        process_fn_map = {
            asr_queue_name: self.asr_message_process,
            llm_queue_name: self.llm_message_process,
        }
        
        # Elegant loop: consume from all queues
        for queue_name, process_fn in process_fn_map.items():
            queue = self.queue_dict[queue_name]
            
            # Create callback that binds the specific process function and queue name
            async def create_callback(fn, qname):
                async def callback(message: aio_pika.IncomingMessage):
                    await self.on_message(fn, qname, message)
                return callback
            
            # Start consuming
            await queue.consume(await create_callback(process_fn, queue_name))
            logger.info(f"{queue_name.upper()} queue consuming")
        
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
        
        