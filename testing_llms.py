import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import re
import logging
import json
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MPS memory limit
if hasattr(torch.mps, 'set_per_process_memory_fraction'):
    torch.mps.set_per_process_memory_fraction(0.7)  # Use 70% of available memory

class MathDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):  # Reduced max_length
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = f"Question: {example['prompt']}\nAnswer: {example['completion']}"
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

class LocalModelTrainer:
    def __init__(self, model_path, num_train_examples=50):  # Reduced number of examples
        self.model_path = model_path
        self.num_train_examples = num_train_examples
        self.metrics_history = {'train': [], 'eval': []}
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Memory cleanup
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def load_and_prepare_model(self):
        """Load model with memory optimizations"""
        logger.info("Loading model...")
        
        # Smaller model configuration for testing
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=1024,  # Reduced size
            intermediate_size=2816,
            num_hidden_layers=16,  # Reduced layers
            num_attention_heads=16,
            num_key_value_heads=16,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,  # Disable KV cache to save memory
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
        )
        
        # Initialize model with gradient checkpointing
        model = LlamaForCausalLM(config)
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        
        # Clear memory
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    def prepare_dataset(self):
        logger.info("Loading dataset...")
        dataset = load_dataset("higgsfield/school-math-questions")
        
        split_name = list(dataset.keys())[0]
        full_dataset = dataset[split_name]
        
        # Take smaller subset for training and evaluation
        train_dataset = full_dataset.select(range(self.num_train_examples))
        eval_dataset = full_dataset.select(range(self.num_train_examples, self.num_train_examples + 10))
        
        # Initialize tokenizer
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create custom datasets
        self.train_dataset = MathDataset(train_dataset, tokenizer, max_length=256)
        self.eval_dataset = MathDataset(eval_dataset, tokenizer, max_length=256)
        
        return self.train_dataset, self.eval_dataset, tokenizer

    def train_model(self, model, train_dataloader, eval_dataloader, num_epochs=10):
        """Memory-optimized training loop"""
        logger.info("Starting training...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        model.to(self.device)
        
        accumulation_steps = 8  # Increased for smaller batches
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear memory
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                
                total_loss += loss.item() * accumulation_steps
                
                # Clear memory periodically
                if i % 10 == 0:
                    gc.collect()
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            
            # Evaluation
            eval_loss = self.evaluate_model(model, eval_dataloader)
            logger.info(f"Epoch {epoch+1} evaluation loss: {eval_loss:.4f}")
            
            self.metrics_history['train'].append(('Epoch ' + str(epoch+1), avg_loss))
            self.metrics_history['eval'].append(('Epoch ' + str(epoch+1), eval_loss))
            
            # Save checkpoint and clear memory
            if (epoch + 1) % 1 == 0:
                checkpoint_dir = f"./checkpoints/epoch_{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                gc.collect()
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        
        return model

    def evaluate_model(self, model, eval_dataloader):
        """Memory-efficient evaluation"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Clear memory
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        
        return total_loss / len(eval_dataloader)

    def train_and_evaluate(self):
        try:
            # Load and prepare model
            model = self.load_and_prepare_model()
            
            # Prepare datasets
            train_dataset, eval_dataset, tokenizer = self.prepare_dataset()
            
            # Create dataloaders with smaller batch size
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=1,  # Minimal batch size
                shuffle=True,
                num_workers=0  # Reduced workers
            )
            
            eval_dataloader = DataLoader(
                eval_dataset, 
                batch_size=1,
                num_workers=0
            )
            
            # Train the model
            model = self.train_model(model, train_dataloader, eval_dataloader)
            
            # Save the final model
            output_dir = "./trained_math_model"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Visualize results
            self.visualize_results()
            
        except RuntimeError as e:
            logger.error(f"Memory error: {str(e)}")
            logger.info("Try reducing model size or batch size further")
            raise
        finally:
            # Clean up memory
            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

    def visualize_results(self):
        # Create DataFrame for visualization
        train_df = pd.DataFrame({
            'Stage': [stage for stage, _ in self.metrics_history['train']],
            'Loss': [loss for _, loss in self.metrics_history['train']],
            'Type': ['Training' for _ in self.metrics_history['train']]
        })
        
        eval_df = pd.DataFrame({
            'Stage': [stage for stage, _ in self.metrics_history['eval']],
            'Loss': [loss for _, loss in self.metrics_history['eval']],
            'Type': ['Evaluation' for _ in self.metrics_history['eval']]
        })
        
        df = pd.concat([train_df, eval_df])
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Stage', y='Loss', hue='Type', marker='o')
        plt.title('Training and Evaluation Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

if __name__ == "__main__":
    model_path = "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
    trainer = LocalModelTrainer(model_path, num_train_examples=50)
    trainer.train_and_evaluate()