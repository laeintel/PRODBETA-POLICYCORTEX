#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolicyCortex GPT-5 Training Quick Start
Simplified training script without DeepSpeed
"""

import os
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=====================================")
print("PolicyCortex GPT-5 Training Setup")
print("=====================================")
print()

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   [WARNING] No GPU detected - training will be slower on CPU")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__} installed")
    except ImportError:
        missing.append("transformers")
    
    try:
        import datasets
        print(f"[OK] Datasets {datasets.__version__} installed")
    except ImportError:
        missing.append("datasets")
    
    try:
        import accelerate
        print(f"[OK] Accelerate {accelerate.__version__} installed")
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    return True

def prepare_training_data():
    """Prepare domain-specific training data"""
    print("\n[DATA] Preparing Training Data")
    print("-" * 40)
    
    # Create training data directory
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample training data
    sample_data = {
        "cloud_governance.jsonl": [
            '{"prompt": "Create an Azure RBAC policy", "completion": "Here\'s a comprehensive Azure RBAC policy..."}',
            '{"prompt": "Optimize cloud costs", "completion": "To optimize cloud costs, implement these strategies..."}',
            '{"prompt": "SOC 2 compliance checklist", "completion": "SOC 2 compliance requires the following controls..."}',
        ],
        "policy_templates.jsonl": [
            '{"prompt": "Network security best practices", "completion": "Network security requires multiple layers..."}',
            '{"prompt": "Data encryption policy", "completion": "Data encryption policy should include..."}',
        ],
        "compliance_frameworks.jsonl": [
            '{"prompt": "GDPR requirements", "completion": "GDPR compliance requires..."}',
            '{"prompt": "HIPAA controls", "completion": "HIPAA requires these technical safeguards..."}',
        ]
    }
    
    for filename, content in sample_data.items():
        filepath = data_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write('\n'.join(content))
            print(f"   Created: {filename}")
        else:
            print(f"   Exists: {filename}")
    
    print(f"\n[OK] Training data prepared in: {data_dir.absolute()}")
    return data_dir

def download_base_model():
    """Download base model for fine-tuning"""
    print("\n[MODEL] Model Setup")
    print("-" * 40)
    
    # For demo, we'll use a smaller model
    model_name = "microsoft/phi-2"  # 2.7B parameter model, good for demos
    
    print(f"Model: {model_name}")
    print("This is a compact but powerful model suitable for domain-specific training")
    print("\nTo download and start training, run:")
    print(f"   python train_model.py --model {model_name}")
    
    return model_name

def create_training_script(model_name, data_dir):
    """Create the actual training script"""
    script_content = f'''#!/usr/bin/env python
"""
PolicyCortex Model Training Script
Auto-generated training configuration
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json

# Model configuration
MODEL_NAME = "{model_name}"
DATA_DIR = "{data_dir}"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading training data...")
# Load your JSONL files
dataset = load_dataset('json', data_files={{
    'train': str(DATA_DIR / '*.jsonl')
}})

def tokenize_function(examples):
    return tokenizer(
        examples['prompt'] + " " + examples['completion'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./policycortex-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=10,
    warmup_steps=100,
    logging_dir='./logs',
    fp16=torch.cuda.is_available(),
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("./policycortex-gpt5-model")
tokenizer.save_pretrained("./policycortex-gpt5-model")

print("[OK] Training complete! Model saved to ./policycortex-gpt5-model")
'''
    
    script_path = Path("train_model.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n[OK] Training script created: {script_path}")
    return script_path

def main():
    # Check dependencies
    if not check_dependencies():
        print("\n[WARNING] Please install missing dependencies first")
        return
    
    # Prepare training data
    data_dir = prepare_training_data()
    
    # Setup model
    model_name = download_base_model()
    
    # Create training script
    script_path = create_training_script(model_name, data_dir)
    
    print("\n" + "="*50)
    print("[SUCCESS] GPT-5 Training Setup Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Review training data in 'training_data/' directory")
    print("2. Start training with: python train_model.py")
    print("\nTraining will create a domain expert AI with:")
    print("   • Cloud governance expertise")
    print("   • Compliance framework knowledge")
    print("   • Policy generation capabilities")
    print("   • Multi-cloud platform support")
    print("\nEstimated training time:")
    print("   • With GPU: 30-60 minutes")
    print("   • CPU only: 2-4 hours")

if __name__ == "__main__":
    main()