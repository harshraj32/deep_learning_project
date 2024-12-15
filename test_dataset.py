
import logging
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaConfig
)
from safetensors.torch import load_file
import json
from tqdm import tqdm
import os
import numpy as np
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, checkpoint_paths):
        self.checkpoint_paths = checkpoint_paths
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def calculate_f1_score(self, predicted, actual):
        try:
            def extract_numbers(text):
                return set(map(float, re.findall(r'-?\d*\.?\d+', text)))
            
            pred_numbers = extract_numbers(predicted)
            actual_numbers = extract_numbers(actual)
            
            if not actual_numbers:
                return 0.0
                
            true_positives = len(pred_numbers.intersection(actual_numbers))
            precision = true_positives / len(pred_numbers) if pred_numbers else 0
            recall = true_positives / len(actual_numbers) if actual_numbers else 0
            
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        except Exception as e:
            logger.warning(f"Error calculating F1 score: {str(e)}")
            return 0.0

    def load_dataset(self):
        logger.info("Loading dataset...")
        dataset = load_dataset("higgsfield/school-math-questions")
        split_name = list(dataset.keys())[0]
        self.eval_data = dataset[split_name].select(range(20))
        logger.info(f"Loaded {len(self.eval_data)} evaluation examples")
        return self.eval_data

    def load_model_and_tokenizer(self, checkpoint_path):
        """Load model and tokenizer from local checkpoint"""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        try:
            # Load configuration
            config_path = os.path.join(checkpoint_path, "config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create configuration object
            config = LlamaConfig(**config_dict)
            logger.info("Created model configuration")
            
            # Initialize empty model
            model = LlamaForCausalLM(config)
            
            # Load the safetensors file
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_path):
                state_dict = load_file(model_path, device="cpu")
                model.load_state_dict(state_dict)
                logger.info("Loaded model weights successfully from safetensors")
            else:
                logger.warning(f"No model weights found at {model_path}")
            
            # Initialize tokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                "hf-internal-testing/llama-tokenizer",
                model_max_length=config.max_position_embeddings,
                padding_side="right",
                use_fast=True,
            )
            
            # Add special tokens
            special_tokens = {
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
            tokenizer.add_special_tokens(special_tokens)
            
            # Resize embeddings if needed
            model.resize_token_embeddings(len(tokenizer))
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {str(e)}")
            raise

    def evaluate_checkpoint(self, checkpoint_path):
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        try:
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer(checkpoint_path)
            model.to(self.device)
            model.eval()
            
            results = []
            total_f1 = 0
            total = 0
            
            with torch.no_grad():
                for example in tqdm(self.eval_data, desc="Evaluating"):
                    question = example["prompt"]
                    actual_answer = example["completion"]
                    
                    # Format input
                    input_text = f"Question: {question}\nAnswer:"
                    inputs = tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    try:
                        # Generate response
                        outputs = model.generate(
                            **inputs,
                            max_length=150,
                            num_return_sequences=1,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        predicted = predicted.split("Answer:")[-1].strip()
                        
                        # Calculate F1 score
                        f1_score = self.calculate_f1_score(predicted, actual_answer)
                        total_f1 += f1_score
                        total += 1
                        
                        results.append({
                            "question": question,
                            "actual": actual_answer,
                            "predicted": predicted,
                            "f1_score": f1_score
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        continue
            
            avg_f1 = total_f1 / total if total > 0 else 0
            logger.info(f"Checkpoint {checkpoint_path} F1 score: {avg_f1:.4f}")
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            return {
                "average_f1": avg_f1,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating checkpoint {checkpoint_path}: {str(e)}")
            return None

    def evaluate_all_checkpoints(self):
        self.load_dataset()
        
        for checkpoint_path in self.checkpoint_paths:
            if not os.path.exists(os.path.join(checkpoint_path, "config.json")):
                logger.error(f"Config not found for checkpoint: {checkpoint_path}")
                continue
                
            if not os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                logger.error(f"Model weights not found for checkpoint: {checkpoint_path}")
                continue
                
            results = self.evaluate_checkpoint(checkpoint_path)
            
            if results:
                self.results[checkpoint_path] = results
                
                # Save results
                output_file = f"{os.path.basename(checkpoint_path)}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {output_file}")
                
                logger.info(f"\nF1 Score: {results['average_f1']:.4f}")

        self.save_comparison_report()

    def save_comparison_report(self):
        """Save a comparison report of all models"""
        comparison = {
            "model_comparison": {
                checkpoint: results["average_f1"]
                for checkpoint, results in self.results.items()
            }
        }
        
        with open("model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("\nModel Comparison Summary:")
        for checkpoint, f1_score in comparison["model_comparison"].items():
            logger.info(f"{os.path.basename(checkpoint)}: F1 = {f1_score:.4f}")

if __name__ == "__main__":
    checkpoint_paths = [
        "checkpoints/epoch_10_llama",
        "checkpoints/epoch_10_qwen",
        "checkpoints/epoch_10_mistral"
    ]
    
    evaluator = ModelEvaluator(checkpoint_paths)
    evaluator.evaluate_all_checkpoints()
