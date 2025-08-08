
# PolicyCortex GPT-5 Training Configuration
MODEL_BASE = "microsoft/phi-2"  # 2.7B parameter model
TRAINING_DATA = "training_data/governance_training.jsonl"
OUTPUT_DIR = "policycortex-gpt5-model"

# Training parameters
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
MAX_LENGTH = 512

# Hardware optimization
USE_FP16 = True  # Half precision for faster training
GRADIENT_CHECKPOINTING = True  # Memory optimization
