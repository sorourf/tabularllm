# TabLLM Benchmarks for Postpartum Depression Classification

This directory contains complete implementations of the TabLLM method (both fine-tuning and few-shot approaches) for postpartum depression classification.

## 📋 Overview

**TabLLM** is a method for applying Large Language Models to tabular classification by converting structured data into natural language text.

**Paper**: [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723)

## 🚀 Quick Start

### Few-Shot Approach (Local)

```bash
# Set API key
export OPENAI_API_KEY='your-key-here'

# Run few-shot evaluation (test on 50 samples first)
python tabllm_fewshot_postpartum.py \
    --data_dir testdata \
    --output_dir tabllm_fewshot_results \
    --num_shots 4 \
    --api_type openai \
    --model gpt-4 \
    --max_samples 50
```

### Fine-Tuning Approach (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `TabLLM_Finetuning_Postpartum_Colab.ipynb`
3. Enable GPU: Runtime → Change runtime type → GPU
4. Upload dataset files or mount Google Drive
5. Run all cells

## 📁 Files

### Implementation Files

| File | Description | Use Case |
|------|-------------|----------|
| `tabllm_fewshot_postpartum.py` | Few-shot learning script | Local execution with API |
| `TabLLM_Finetuning_Postpartum_Colab.ipynb` | Fine-tuning notebook | Google Colab training |
| `compare_tabllm_results.py` | Comparison script | Compare both approaches |

### Documentation

| File | Description |
|------|-------------|
| `TABLLM_IMPLEMENTATION_GUIDE.md` | **Comprehensive guide** (start here!) |
| `README_TABLLM_BENCHMARKS.md` | This file - quick reference |
| `TABLLM_POSTPARTUM_DEPRESSION_README.md` | Baseline results (TF-IDF) |

### Existing Implementations

| File | Description |
|------|-------------|
| `evaluate_postpartum_depression.py` | Baseline TF-IDF + Logistic Regression |
| `helper/external_datasets_variables.py` | Template definitions |
| `helper/note_template.py` | Template engine |

## 📊 Comparison: Few-Shot vs Fine-Tuning

| Aspect | Few-Shot | Fine-Tuning |
|--------|----------|-------------|
| **Setup** | 5 min | 15 min |
| **Training** | None | 30-60 min |
| **GPU Needed** | No | Yes |
| **Cost** | $2-6 (API) | Free (Colab) |
| **Accuracy** | 70-85% | 75-90% |
| **Best For** | Quick experiments | Production |

## 🎯 Expected Performance

Based on similar medical classification tasks:

### Few-Shot Learning
- **Accuracy**: 70-85%
- **F1-Score**: 75-88%
- **AUC-ROC**: 80-90%

### Fine-Tuning
- **Accuracy**: 75-90%
- **F1-Score**: 78-92%
- **AUC-ROC**: 85-95%

### Baseline (TF-IDF)
- **Accuracy**: 78.79%
- **F1-Score**: 82.18%
- **AUC-ROC**: 89.06%

## 📖 Detailed Documentation

See **[TABLLM_IMPLEMENTATION_GUIDE.md](TABLLM_IMPLEMENTATION_GUIDE.md)** for:

- ✅ Complete setup instructions
- ✅ Step-by-step tutorials
- ✅ Troubleshooting guide
- ✅ Best practices
- ✅ Clinical implications
- ✅ Advanced topics

## 🔧 Installation

### For Few-Shot Approach

```bash
pip install pandas numpy scikit-learn openai anthropic
```

### For Fine-Tuning (Local)

```bash
pip install transformers==4.30.0 datasets==2.14.0 accelerate==0.20.0
pip install torch torchvision torchaudio
pip install sentencepiece protobuf scikit-learn pandas numpy
```

### For Comparison Script

```bash
pip install matplotlib seaborn
```

## 🧪 Testing the Implementations

### Test Few-Shot (No API calls)

```bash
# This will fail at API call, but validates data loading
python tabllm_fewshot_postpartum.py --max_samples 5 --api_type local
```

### Test Fine-Tuning (Locally)

See notebook cell-by-cell execution instructions in `TABLLM_IMPLEMENTATION_GUIDE.md`

### Compare Results

```bash
python compare_tabllm_results.py \
    --fewshot_dir tabllm_fewshot_results \
    --finetuning_dir tabllm_finetuned \
    --output_dir tabllm_comparison \
    --create_plots
```

## 📦 Dataset

**Location**: `testdata/`
- `train_postpartum_depression.csv` (1,043 samples)
- `test_postpartum_depression.csv` (448 samples)

**Features**: 9 maternal health indicators
**Target**: Feeling anxious (Yes/No) - predictor of postpartum depression

See [TABLLM_IMPLEMENTATION_GUIDE.md](TABLLM_IMPLEMENTATION_GUIDE.md#dataset-description) for details.

## 🎓 Usage Examples

### Example 1: Quick Prototype with Few-Shot

```bash
# Test with 50 samples using GPT-3.5 (cheaper)
python tabllm_fewshot_postpartum.py \
    --num_shots 4 \
    --api_type openai \
    --model gpt-3.5-turbo \
    --max_samples 50 \
    --output_dir quick_test
```

### Example 2: Full Fine-Tuning

```python
# In Colab notebook, adjust these parameters:
MODEL_NAME = "t5-base"
num_train_epochs = 10
per_device_train_batch_size = 8
learning_rate = 5e-4
```

### Example 3: Ensemble Both Approaches

```python
# After running both approaches
import pandas as pd
import numpy as np

fewshot = pd.read_csv('tabllm_fewshot_results/fewshot_predictions.csv')
finetuned = pd.read_csv('tabllm_finetuned/finetuning_predictions.csv')

# Simple voting ensemble
ensemble = (fewshot['predicted_binary'] + finetuned['predicted_binary']) >= 1

# Evaluate ensemble
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(fewshot['true_binary'], ensemble)
print(f"Ensemble Accuracy: {accuracy:.4f}")
```

## ⚠️ Important Notes

### API Costs (Few-Shot)
- **Full test set (448 samples)**: $2-6
- **Test first with `--max_samples 50`** to verify setup

### GPU Requirements (Fine-Tuning)
- **Minimum**: Google Colab T4 (free tier)
- **Recommended**: V100 or A100 for larger models
- **Model size**: t5-base works on T4, t5-large needs V100+

### Clinical Use
⚠️ **These models are for screening, not diagnosis**
- Use as initial risk assessment
- Combine with clinical evaluation
- Validate with domain experts before deployment

## 🐛 Troubleshooting

### Few-Shot Issues

**"API Key not found"**
```bash
export OPENAI_API_KEY='sk-...'
# Verify:
echo $OPENAI_API_KEY
```

**"Rate limit exceeded"**
```bash
# Increase delay between calls
python tabllm_fewshot_postpartum.py --batch_delay 2.0
```

### Fine-Tuning Issues

**"Out of memory"**
- Reduce batch size to 4
- Use t5-base instead of t5-large
- Enable GPU in Colab

**"No GPU available"**
- Colab: Runtime → Change runtime type → GPU
- Verify with `!nvidia-smi`

See full troubleshooting guide in [TABLLM_IMPLEMENTATION_GUIDE.md](TABLLM_IMPLEMENTATION_GUIDE.md#troubleshooting)

## 📈 Results Structure

### Few-Shot Output

```
tabllm_fewshot_results/
├── fewshot_metrics.json          # Evaluation metrics
└── fewshot_predictions.csv       # Per-sample predictions
```

### Fine-Tuning Output

```
tabllm_finetuned/
├── final_model/                  # Trained model
├── finetuning_metrics.json       # Evaluation metrics
└── finetuning_predictions.csv    # Per-sample predictions
```

### Comparison Output

```
tabllm_comparison/
├── comparison_report.md          # Comprehensive comparison
├── metrics_comparison.csv        # Metrics table
├── agreement_stats.json          # Agreement analysis
├── metrics_comparison.png        # Bar plot
└── agreement_analysis.png        # Pie chart
```

## 🔬 Methodology

### Text Serialization

TabLLM converts each tabular row to natural language:

**Input (Tabular)**:
```
age: 40-45, feeling_sad: Yes, irritable: No, ...
```

**Output (Text)**:
```
- age range: 40-45
- feeling sad or tearful: Yes
- irritable towards baby and partner: No
- trouble sleeping at night: Two or more days a week
...
```

### Few-Shot Prompt

```
You are a medical AI assistant...

Patient Profile:
- age range: 30-35
- feeling sad or tearful: Sometimes
...
Feeling Anxious: Yes

[3 more examples]

Now classify this patient:
[New patient profile]
Feeling Anxious:
```

### Fine-Tuning

1. Serialize all training data to text
2. Fine-tune T5 model on (text, label) pairs
3. Use parameter-efficient methods (similar to IA3)
4. Evaluate on serialized test data

## 📚 References

### Papers
- [TabLLM](https://arxiv.org/pdf/2210.10723) - Main paper
- [T5](https://arxiv.org/abs/1910.10683) - T5 model
- [T-Few](https://arxiv.org/abs/2205.05638) - Few-shot fine-tuning

### Repositories
- [TabLLM GitHub](https://github.com/clinicalml/TabLLM)
- [T-Few GitHub](https://github.com/r-three/t-few)

### Documentation
- [OpenAI API](https://platform.openai.com/docs)
- [Hugging Face](https://huggingface.co/docs)

## 🤝 Contributing

These implementations are part of a research project on applying TabLLM to medical classification tasks.

## 📄 Citation

```bibtex
@inproceedings{hegselmann2023tabllm,
  title={Tabllm: Few-shot classification of tabular data with large language models},
  author={Hegselmann, Stefan and Buendia, Alejandro and Lang, Hunter and Agrawal, Monica and Jiang, Xiaoyi and Sontag, David},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={5549--5581},
  year={2023},
  organization={PMLR}
}
```

## 📞 Support

For questions:
1. Check [TABLLM_IMPLEMENTATION_GUIDE.md](TABLLM_IMPLEMENTATION_GUIDE.md)
2. Review original [TabLLM repository](https://github.com/clinicalml/TabLLM)
3. See [Troubleshooting section](#-troubleshooting)

---

**Last Updated**: November 2025
**Version**: 1.0
**Dataset**: Postpartum Depression Classification
**Implementation**: TabLLM Few-Shot + Fine-Tuning
