import logging
import os

os.makedirs("log", exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log/app_test_take_off.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger():
    return logger