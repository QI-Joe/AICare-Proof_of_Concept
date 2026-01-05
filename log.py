import logging
import os
import datetime

today = str(datetime.date.today())
os.makedirs("log", exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"log/app_test_{today}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger():
    return logger