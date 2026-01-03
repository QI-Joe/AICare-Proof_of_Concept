import pika
import json
from typing import List, Optional, Dict, Any
from ASR_model import ASRModel
import base64
import numpy as np
from log import get_logger

logger = get_logger()


class ASRServer:
    def __init__(self):
        self.asr_model = ASRModel("openai-whisper-small")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host = 'localhost',
                heartbeat=200,
                blocked_connection_timeout=300,
                connection_attempts=3,
                retry_delay=2,
                )
            )
        self.channel = self.connection.channel()
        
        # Check and log model status
        status = self.asr_model.check_model_status()
        logger.info("=" * 60)
        logger.info("ASR Model Status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def run(self):
        self.channel.exchange_declare(exchange="asr_task_exchange", exchange_type="direct", durable=True)
        
        # what is relations of queue and exchange, how it could be used?
        self.channel.queue_declare(queue="asr_queue", durable=True)
        
        # Purge old messages from queue (useful during development)
        purged = self.channel.queue_purge(queue="asr_queue")
        logger.info(f"Purged {purged} old messages from queue")
        print(f"Purged {purged} old messages from queue")
        
        def callback(ch, method, props, body):
            task_info = json.loads(body)
            audio_data: List[float] = task_info["audio"]
            audio_data = np.frombuffer(
                base64.b64decode(
                    audio_data
                    ),
                dtype=np.float32
                )
            
            sample_rate: float = task_info['sample_rate']
            recognized_text = self.asr_model.process(audio_data, sample_rate)
            result = {
                "text": recognized_text
            }
            ch.basic_publish(
                exchange='',
                routing_key = props.reply_to,
                properties = pika.BasicProperties(
                   correlation_id=props.correlation_id
                ), 
                body = json.dumps(result)
            )
            ch.basic_ack(delivery_tag = method.delivery_tag)
            
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue="asr_queue", on_message_callback=callback)
        print("Server side start listening...")
        self.channel.start_consuming()

if __name__ == "__main__":
    asr_backend = ASRServer()
    asr_backend.run()
        
        