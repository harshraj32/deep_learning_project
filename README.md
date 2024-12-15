# Educational LLM RAG Bot

An educational application that evaluates and compares different Large Language Models (LLMs) on mathematical problem-solving tasks. The project implements a complete pipeline for training, evaluation, and comparison of LLaMA, Qwen, and Mistral models.

## Overview

This project provides:
- Custom training pipeline for fine-tuning LLMs on mathematical problems
- Comprehensive evaluation framework for model comparison
- Memory-optimized implementation for resource-constrained environments
- Visualization tools for performance analysis

## Models Used

- [LLaMA 3.2B Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) - Efficient 3B parameter model
- [Qwen 7B Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/tree/main) - High-performance 7B parameter model
- [Mistral 7B Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main) - Advanced 7B parameter model
- Dataset: [School Math Questions](https://huggingface.co/datasets/higgsfield/school-math-questions)

## Project Structure

```
.
├── test_dataset.py        # Model evaluation implementation
├── testing_llms.py        # Training and fine-tuning pipeline
├── llms_visuals.py        # Visualization utilities
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/educational-llm-math-solver.git
cd educational-llm-math-solver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
```bash
# Run the download script (if provided) or manually download from HuggingFace
python download_models.py
```

## Usage

### Training
```python
from testing_llms import LocalModelTrainer

# Initialize trainer
trainer = LocalModelTrainer(
    model_path="path_to_model",
    num_train_examples=50
)

# Train and evaluate
trainer.train_and_evaluate()
```

### Evaluation
```python
from test_dataset import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator([
    "checkpoints/epoch_10_llama",
    "checkpoints/epoch_10_qwen",
    "checkpoints/epoch_10_mistral"
])

# Run evaluation
evaluator.evaluate_all_checkpoints()
```

### Visualization
```python
from llms_visuals import visualize_results

# Generate visualization
visualize_results()
```
## Final Run
```bash
streamlit run app.py
```

## Results

Model performance comparison (F1 Scores):

| Model | F1 Score |
|-------|----------|
| Qwen-7B-Instruct | 0.6255 |
| LLaMA-3.2B-Instruct | 0.5371 |
| Mistral-7B-Instruct | 0.2531 |

## Features

- **Memory Optimization**:
  - Gradient checkpointing
  - Dynamic memory cleanup
  - Device-specific optimizations

- **Evaluation Metrics**:
  - F1 score calculation
  - Response accuracy assessment
  - Processing speed monitoring

- **Visualization Tools**:
  - Training progress plots
  - Model comparison charts
  - Performance metrics visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Additional requirements in `requirements.txt`

## Performance Optimization

The implementation includes several optimization techniques:

```python
# Memory management
if hasattr(torch.mps, 'set_per_process_memory_fraction'):
    torch.mps.set_per_process_memory_fraction(0.7)

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Batch size optimization
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=1,
    shuffle=True,
    num_workers=0
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace for model and dataset hosting
- PyTorch team for the deep learning framework
- Original model creators (LLaMA, Qwen, Mistral)
