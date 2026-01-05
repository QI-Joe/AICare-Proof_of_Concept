import aio_pika
import asyncio
import uuid
import json
import base64, log
import numpy as np
from typing import Optional, Dict, Callable, Any

logger = log.get_logger()

class RPClient:
    def __init__(self):
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.callback_queue: Optional[aio_pika.Queue] = None
        self.futures: Dict[str, asyncio.Future] = dict()
        self.consumer_tag: Optional[str] = None
        self.params = {
            "host": "localhost",
            "login": "guest",
            "password": "guest",
            "heartbeat": 60,
        }
        
    async def _on_response(self, message: aio_pika.IncomingMessage):
        """Handle RPC responses"""
        logger.info("Start to listening processed feedback from Worker...")
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id in self.futures:
                future = self.futures.pop(correlation_id)
                result = json.loads(message.body)
                future.set_result(result)
        
    async def connect(self):
        self.connection = await aio_pika.connect_robust(**self.params)
        self.channel = await self.connection.channel()
        
        self.callback_queue = await self.channel.declare_queue('', exclusive=True, auto_delete=True)
        self.consumer_tag = await self.callback_queue.consume(self._on_response)
    
    def corrid_upload(self):
        correlation_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.futures[correlation_id] = future
        return correlation_id, future
    
    async def call_asr(self, audio_array: np.ndarray, sample_rate: int, 
                      timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send audio for ASR processing
        """
        correlation_id, future = self.corrid_upload()
        # Prepare request
        request_data = {
            "audio": base64.b64encode(audio_array.tobytes()).decode('utf-8'),
            "sample_rate": sample_rate,
            "data_type": "np.float32",
        }
        
        # Publish request
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(request_data).encode(),
                content_type="application/json",
                correlation_id=correlation_id,
                reply_to=self.callback_queue.name,
            ),
            routing_key="asr_queue",
        )
        
        logger.debug(f"Sent ASR request {correlation_id[:8]}...")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.futures.pop(correlation_id, None)
            raise TimeoutError(f"ASR request timed out after {timeout}s")
        
    async def call_llm(self, text: str, timeout:float = 30.0) -> Dict[str, Any]:
        correlation_id, future = self.corrid_upload()
        request_data = {
            'text': text,
        }
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body = json.dumps(request_data).encode(),
                content_type='application/json',
                correlation_id=correlation_id,
                reply_to=self.callback_queue,
            ),
            routing_key='llm_queue',
        )
        
        logger.debug(f"Sent LLM request {correlation_id[:8]}...")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.futures.pop(correlation_id, None)
            raise TimeoutError(f"ASR request timed out after {timeout}s")
        
    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()
            logger.info("RPC Client connection closed")

