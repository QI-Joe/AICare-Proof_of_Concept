import asyncio
import websockets
import json
import pika
import uuid
from typing import Dict, Optional, Any
import threading
from log import get_logger


"""
Dec 5th working dialogue
FINISHED:
    1. Entire process + fundmental calling stream
    2. Asyncio test set / log record added

TODO: 
    1. Stream transmission + real - scenoiro test
    2. Concept of redditmq and websocket mechnism, also a small front-end
    3. aio-pika is a library for asyncio pika, a new technique may used in future
    4. think about a question: what if in server side there are multi models gonna run? how should we allocate queue and bind with exchange?
"""

logger = get_logger()


class ASRProducer:
    def __init__(self):
        self.__loop: Optional[Any | asyncio.events.AbstractEventLoop] = None
        self.__socket_dict = dict()
        self.callback_queue_name: Any = None
        
        self.params = pika.ConnectionParameters(
            host = "localhost", 
            heartbeat=200, 
            blocked_connection_timeout=300, 
            connection_attempts=3
        )
        
        self.build_consumer()
        self.build_publisher()
        
    def loop_inject(self, loop: asyncio.events.AbstractEventLoop):
        self.__loop = loop
        
    def get_dict_len(self):
        return len(self.__socket_dict)
    
    def add_new_map(self, corrid: str, future: asyncio.futures.Future):
        self.__socket_dict[corrid] = future
    
    def build_publisher(self):
        self.publish_connection = pika.BlockingConnection(self.params)
        self.publish_channel = self.publish_connection.channel()
        self.secure_lock = threading.Lock()
        
    def build_consumer(self):
        self.consumer_connection = pika.BlockingConnection(self.params)
        self.consumer_channel = self.consumer_connection.channel()
        
    def listening_on_feedback(self):
        # Declare exclusive, auto-delete queue for RPC responses
        result_queue = self.consumer_channel.queue_declare(
            queue='', 
            exclusive=True,  # Queue deleted when connection closes
            auto_delete=True  # More appropriate for callback queues
        )
        self.callback_queue_name = result_queue.method.queue
        logger.info(f"Callback queue created: {self.callback_queue_name}")
        print(f"Callback queue created: {self.callback_queue_name}")
        
        def on_response(ch, method, props, body):
            c_id = props.correlation_id
            if c_id in self.__socket_dict:
                map_future = self.__socket_dict.pop(c_id)
                # Fix: set_result (not set_results), body is already bytes
                self.__loop.call_soon_threadsafe(map_future.set_result, body)
            else:
                logger.warning(f"Received response for unknown correlation_id: {c_id}")
        
        self.consumer_channel.basic_consume(
            queue=self.callback_queue_name, on_message_callback=on_response, auto_ack=True,
        )
        logger.info("Pika pending for processed reuslt")
        print("Pika pending for processed reuslt")
        self.consumer_channel.start_consuming()
        
    def publish_client_task(self, message, corr_id):
        # Ensure message is bytes or string, not dict or other object
        if isinstance(message, dict):
            message = json.dumps(message)
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        with self.secure_lock:
            self.publish_channel.basic_publish(
                exchange='', # default exchange
                routing_key='asr_queue',
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue_name,
                    correlation_id=corr_id,
                ),
                body=message,
            )

async def websocket_handler(websocket):
    try:
        async for message in websocket:
            logger.info("get user data")
            print("get user data")
            corr_id = str(uuid.uuid4())
            
            current_loop = asyncio.get_running_loop()
            future = current_loop.create_future()
            
            asr_websocket.loop_inject(current_loop)
            asr_websocket.add_new_map(corr_id, future)
            logger.info(f"📤 [Send] correlation_id: {corr_id[:8]}..., pending tasks: {asr_websocket.get_dict_len()}")
            print(f"📤 [Send] correlation_id: {corr_id[:8]}..., pending tasks: {asr_websocket.get_dict_len()}")
            
            # Key step: Send to RabbitMQ in thread pool (prevent blocking WebSocket)
            await current_loop.run_in_executor(
                None, 
                asr_websocket.publish_client_task, 
                message, 
                corr_id
            )
            
            # Wait for result (await suspends current task until Future is set)
            result = await future
            logger.info(f"📥 [Received] correlation_id: {corr_id[:8]}..., remaining: {asr_websocket.get_dict_len()}")
            print(f"📥 [Received] correlation_id: {corr_id[:8]}..., remaining: {asr_websocket.get_dict_len()}")
            
            # Ensure result is properly formatted as string for WebSocket
            if isinstance(result, bytes):
                result = result.decode('utf-8')
            elif isinstance(result, dict):
                logger.error("Got warning, data is a dict-like object")
                result = json.dumps(result)
            
            await websocket.send(result)
    except Exception as e:
        logger.error(f"connection handler failed")
        logger.error(str(e))
        raise Exception(e)

# 启动WebSocket服务器
async def main():
    # Set max_size to handle large audio payloads (default is 1MB, set to 10MB)
    # Set max_queue to limit memory usage per connection
    async with websockets.serve(
        websocket_handler, 
        "0.0.0.0", 
        8765,
        max_size=10 * 1024 * 1024,  # 10MB max message size
        max_queue=32,  # Limit queued messages
        ping_interval=20,  # Keep connection alive
        ping_timeout=20
    ):
        logger.info("server listening on 0.0.0.0:8765")
        print("[网关] WebSocket网关已在 ws://0.0.0.0:8765 启动")
        await asyncio.Future() 
        

if __name__ == "__main__":
    asr_websocket = ASRProducer()
    listen_thread = threading.Thread(target=asr_websocket.listening_on_feedback, daemon=True)
    listen_thread.start()  # Fix: Start the listener thread!
    logger.info("RabbitMQ listener thread started")
    print("RabbitMQ listener thread started")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        print("\nServer stopped by user")
    finally:
        asr_websocket.publish_connection.close()
        asr_websocket.consumer_connection.close()
        logger.info("Connections closed")
        print("Connections closed")
