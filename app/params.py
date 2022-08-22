from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs/fit') / datetime.now().strftime("%Y%m%d-%H%M%S")

BACHSIZE = 500
EPOCHS = 50
VALIDATION_SPLIT = 0.2

INPUT_SIZE = 784
HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 128
OUTPUT_SIZE = 10

MODEL_FILE_PATH = 'model.h5'
