# DeepSeek-R1 14B Model

## Overview
DeepSeek-R1 14B is an open-source large language model (LLM) designed to enhance reasoning capabilities using reinforcement learning (RL). This model builds upon the DeepSeek-R1-Zero foundation and integrates multi-stage training techniques, including cold-start data and rejection sampling, to optimize reasoning performance. It achieves state-of-the-art results in various benchmarks, making it a powerful tool for AI research and applications.

## Key Features
- **Model Variants:** Available in multiple sizes, including 1.5B, 7B, 8B, 14B, 32B, and 70B.
- **Training Methodology:** Utilizes reinforcement learning with reasoning-oriented fine-tuning.
- **Performance:** Competitive with OpenAI models on reasoning tasks like MATH-500, Codeforces, and GPQA Diamond.
- **Distillation:** Smaller versions inherit reasoning abilities from larger models.
- **Benchmarks:** Strong performance across reasoning, mathematical, and software engineering tasks.

## Installation
Ensure you have the necessary dependencies installed before loading the model.

```bash
pip install torch transformers accelerate bitsandbytes
```

## Model Usage
### Loading the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-14b")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-r1-14b", device_map="auto")
```

### Generating Responses
```python
input_text = "What are the key advancements in reinforcement learning?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Benchmark Performance
| Benchmark       | DeepSeek-R1 14B | OpenAI-o1-1217 |
|----------------|---------------|---------------|
| AIME 2024 (Pass@1) | 79.8% | 79.2% |
| Codeforces (Percentile) | 96.3% | 96.6% |
| MATH-500 (Pass@1) | 97.3% | 96.4% |
| MMLU (Pass@1) | 90.8% | 91.8% |
| SWE-bench Verified (Resolved) | 49.2% | 48.9% |

## Repository
Find the implementation and additional details on GitHub:
🔗 [DeepSeek-R1 14B Repository](https://github.com/Uddeshya8272/Deepseek_R1_14b)

## Citations
If you use DeepSeek-R1 in your research, please cite:
```
@article{deepseek2025,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek AI Team},
  year={2025}
}
```

## License
DeepSeek-R1 14B is released under an open-source license for research and development purposes.

