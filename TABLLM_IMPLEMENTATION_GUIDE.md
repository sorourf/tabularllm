# TabLLM Implementation Guide for Postpartum Depression Classification

This guide provides comprehensive instructions for implementing TabLLM (both fine-tuning and few-shot approaches) on the postpartum depression dataset.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Approach 1: Few-Shot Learning (Local)](#approach-1-few-shot-learning-local)
4. [Approach 2: Fine-Tuning (Google Colab)](#approach-2-fine-tuning-google-colab)
5. [Comparison of Approaches](#comparison-of-approaches)
6. [Results and Evaluation](#results-and-evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Overview

**TabLLM** is a method for applying Large Language Models (LLMs) to tabular classification tasks by converting structured data into natural language text using domain-aware templates.

**Paper**: [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723)

**Two Main Approaches**:
1. **Few-Shot Learning**: Uses LLM APIs (OpenAI, Anthropic) with few examples in the prompt
2. **Fine-Tuning**: Fine-tunes a T5-based model on the serialized training data

---

## Dataset Description

**Dataset**: Postpartum Depression Classification
**Source**: Medical hospital questionnaire (Google Form)
**Total Records**: 1,503 samples
**Training Samples**: 1,043
**Test Samples**: 448

### Features (9 total)

1. **Age**: Age range (25-30, 30-35, 35-40, 40-45, 45-50)
2. **Feeling sad or tearful**: Yes/No/Sometimes
3. **Irritable towards baby & partner**: Yes/No/Sometimes
4. **Trouble sleeping at night**: Yes/No/Two or more days a week
5. **Problems concentrating or making decision**: Yes/No/Often
6. **Overeating or loss of appetite**: Yes/No/Not at all
7. **Feeling of guilt**: Yes/No/Maybe
8. **Problems of bonding with baby**: Yes/No/Sometimes
9. **Suicide attempt**: Yes/No/Not interested to say

### Target Variable

**Feeling anxious**: Binary classification (Yes/No) - predictor of postpartum depression

**Class Distribution**:
- Training: 677 anxious (64.9%), 366 not anxious (35.1%)
- Test: 291 anxious (64.9%), 157 not anxious (35.1%)

---

## Approach 1: Few-Shot Learning (Local)

### Overview

The few-shot approach uses LLM APIs (OpenAI GPT-4, Claude, etc.) to classify samples based on a small number of examples included in the prompt.

### Advantages

- ✅ No training required
- ✅ Fast to implement
- ✅ Works well with limited data
- ✅ Can leverage powerful pre-trained models

### Disadvantages

- ❌ Requires API access (costs money)
- ❌ Rate limits may slow down evaluation
- ❌ Limited to API's context window
- ❌ No model ownership

### Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn openai anthropic

# Set API key (choose one)
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

### Usage

#### Basic Usage

```bash
python tabllm_fewshot_postpartum.py \
    --data_dir testdata \
    --output_dir tabllm_fewshot_results \
    --num_shots 4 \
    --api_type openai \
    --model gpt-4
```

#### Arguments

- `--data_dir`: Directory containing train/test CSV files (default: `testdata`)
- `--output_dir`: Directory to save results (default: `tabllm_fewshot_results`)
- `--num_shots`: Number of examples in prompt (default: `4`)
- `--api_type`: API type - `openai`, `anthropic`, or `local` (default: `openai`)
- `--model`: Model name (default: `gpt-4`)
- `--seed`: Random seed for example selection (default: `42`)
- `--batch_delay`: Delay between API calls in seconds (default: `1.0`)
- `--max_samples`: Maximum test samples to evaluate (default: `None` for all)

#### Example: Quick Test with 50 Samples

```bash
python tabllm_fewshot_postpartum.py \
    --data_dir testdata \
    --output_dir tabllm_fewshot_test \
    --num_shots 4 \
    --api_type openai \
    --model gpt-4 \
    --max_samples 50
```

#### Example: Using Claude (Anthropic)

```bash
python tabllm_fewshot_postpartum.py \
    --data_dir testdata \
    --output_dir tabllm_fewshot_claude \
    --num_shots 8 \
    --api_type anthropic \
    --model claude-3-opus-20240229
```

### Output Files

The script creates the following files in the output directory:

1. **fewshot_metrics.json**: Comprehensive evaluation metrics
2. **fewshot_predictions.csv**: Per-sample predictions with LLM responses
3. **Console output**: Real-time progress and results

### Expected Performance

Based on the TabLLM paper and similar medical classification tasks:

- **Accuracy**: 70-85%
- **F1-Score**: 75-88%
- **AUC-ROC**: 80-90%

Performance varies by:
- Number of shots (more shots generally better, but diminishing returns after 8-16)
- Model quality (GPT-4 > GPT-3.5, Claude Opus > Claude Sonnet)
- Example selection strategy

### Cost Estimation

**For full test set (448 samples)**:

- **GPT-4**: ~$2-5 (depending on context length)
- **GPT-3.5**: ~$0.50-1
- **Claude Opus**: ~$3-6
- **Claude Sonnet**: ~$1-2

Costs can be reduced by:
- Using `--max_samples` to test on subset first
- Using cheaper models (GPT-3.5, Claude Sonnet)
- Reducing `--num_shots`

---

## Approach 2: Fine-Tuning (Google Colab)

### Overview

The fine-tuning approach trains a T5-based model on the serialized tabular data using parameter-efficient methods.

### Advantages

- ✅ No API costs after training
- ✅ Full control over model
- ✅ Can be deployed offline
- ✅ Potentially better performance with more data

### Disadvantages

- ❌ Requires GPU for training (use Google Colab free tier)
- ❌ Takes longer (30-60 minutes training)
- ❌ Requires more technical knowledge
- ❌ Need to manage model storage

### Prerequisites

**For Google Colab**:
- Google account
- Dataset files ready to upload

**For Local Execution**:
```bash
pip install transformers==4.30.0 datasets==2.14.0 accelerate==0.20.0
pip install torch torchvision torchaudio
pip install sentencepiece protobuf scikit-learn pandas numpy
```

### Usage

#### Step-by-Step Guide

1. **Open Google Colab**
   - Go to [https://colab.research.google.com](https://colab.research.google.com)
   - Upload the notebook: `TabLLM_Finetuning_Postpartum_Colab.ipynb`

2. **Enable GPU**
   - Click: Runtime → Change runtime type
   - Hardware accelerator: GPU (T4 recommended)
   - Click Save

3. **Upload Dataset**
   - Option A: Use file upload in Colab
   - Option B: Upload to Google Drive and mount

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Run All Cells**
   - Click: Runtime → Run all
   - Or run cells sequentially (recommended for first time)

5. **Monitor Training**
   - Training typically takes 30-60 minutes
   - Watch for training/eval loss decreasing
   - GPU memory usage should stay under limit

6. **Download Results**
   - Results are saved to `/content/tabllm_finetuned/`
   - Download the zip file created in final cell
   - Or copy to Google Drive

### Configuration Options

Edit these in the notebook:

```python
# Model selection
MODEL_NAME = "t5-base"  # Options: "t5-base", "t5-large", "google/t5-v1_1-base"

# Training hyperparameters
num_train_epochs = 10       # Number of epochs (3-15 typical)
per_device_train_batch_size = 8   # Batch size (4-16 depending on GPU)
learning_rate = 5e-4        # Learning rate (1e-5 to 1e-3)
```

### Hardware Requirements

| Model | GPU | RAM | Training Time |
|-------|-----|-----|---------------|
| t5-base | T4 | Standard | 30-45 min |
| t5-large | V100 | High-RAM | 60-90 min |
| t5-3b | A100 | High-RAM | 2-3 hours |

**Free Tier**: T4 GPU with standard RAM works well with `t5-base`

### Output Files

1. **final_model/**: Trained model and tokenizer
2. **finetuning_metrics.json**: Evaluation metrics
3. **finetuning_predictions.csv**: Per-sample predictions
4. **logs/**: Training logs
5. **tabllm_results.zip**: All results packaged

### Expected Performance

Based on similar medical classification tasks with TabLLM:

- **Accuracy**: 75-90%
- **F1-Score**: 78-92%
- **AUC-ROC**: 85-95%

Performance improvements:
- Using `t5-large` instead of `t5-base`: +2-5% accuracy
- More training epochs (up to point of overfitting)
- Hyperparameter tuning

### Hyperparameter Tuning Tips

1. **Learning Rate**
   - Too high: Loss oscillates, doesn't converge
   - Too low: Slow training, may not reach optimal
   - Good range: 1e-4 to 1e-3 for T5

2. **Batch Size**
   - Larger: More stable gradients, but uses more memory
   - Smaller: More updates per epoch, but noisier
   - Adjust based on GPU memory

3. **Epochs**
   - Monitor eval loss to detect overfitting
   - Early stopping when eval loss stops improving
   - Typical range: 5-15 epochs

---

## Comparison of Approaches

| Aspect | Few-Shot | Fine-Tuning |
|--------|----------|-------------|
| **Setup Time** | 5 minutes | 10-15 minutes |
| **Training Time** | None | 30-60 minutes |
| **Inference Time** | 1-2 sec/sample | 0.01-0.1 sec/sample |
| **Cost** | API costs ($2-6) | Free (Colab) or GPU costs |
| **GPU Required** | No | Yes (for training) |
| **Accuracy** | 70-85% | 75-90% |
| **Customization** | Limited | High |
| **Offline Use** | No | Yes (after training) |
| **Best For** | Quick experiments, small datasets | Production, large datasets |

### When to Use Each

**Use Few-Shot When**:
- Quick prototyping or proof-of-concept
- Small evaluation sets (<100 samples)
- Don't have GPU access
- Want to try multiple models easily
- Need flexibility in prompt engineering

**Use Fine-Tuning When**:
- Need fast inference on many samples
- Want to deploy offline/in production
- Have GPU access (free via Colab)
- Want full model control
- Need reproducible results

---

## Results and Evaluation

### Metrics Reported

Both approaches report:

1. **Accuracy**: Overall correctness
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under ROC curve
6. **Confusion Matrix**: Breakdown of predictions

### Interpreting Results for Medical Task

**For postpartum depression screening**:

- **High Precision** (>85%): Important to minimize false alarms
- **High Recall** (>75%): Important to catch most at-risk patients
- **Balanced F1**: Overall effectiveness
- **AUC-ROC >80%**: Good discrimination ability

**Clinical Implications**:
- These models are for **screening**, not diagnosis
- Should be combined with clinical assessment
- Consider threshold adjustment based on use case:
  - **High Recall**: Screening tool (catch all potential cases)
  - **High Precision**: Triage tool (prioritize high-confidence cases)

### Comparing Results

To compare few-shot vs fine-tuning results:

```bash
# Run comparison script (create after both evaluations)
python compare_results.py \
    --fewshot_dir tabllm_fewshot_results \
    --finetuning_dir tabllm_finetuned
```

---

## Troubleshooting

### Few-Shot Approach

**Problem**: API Key Error
```
Solution: Ensure API key is set correctly
export OPENAI_API_KEY='your-key-here'
# Verify:
echo $OPENAI_API_KEY
```

**Problem**: Rate Limit Exceeded
```
Solution: Increase --batch_delay
python tabllm_fewshot_postpartum.py --batch_delay 2.0
```

**Problem**: Poor Predictions (Random Guessing)
```
Possible causes:
1. Wrong API key
2. Insufficient shots (try --num_shots 8)
3. Model too small (try GPT-4 instead of GPT-3.5)
```

### Fine-Tuning Approach

**Problem**: Out of Memory (OOM)
```
Solutions:
1. Reduce batch size (try 4 instead of 8)
2. Use gradient accumulation
3. Use smaller model (t5-base instead of t5-large)
4. Enable GPU in Colab settings
```

**Problem**: Training Very Slow
```
Solutions:
1. Verify GPU is enabled (Runtime → Change runtime type)
2. Check GPU utilization (watch -n 1 nvidia-smi)
3. Reduce max_length in tokenizer
```

**Problem**: Model Not Improving
```
Solutions:
1. Increase learning rate (try 1e-3)
2. More epochs (try 15-20)
3. Check data preprocessing
4. Verify labels are correct
```

**Problem**: Predictions All Same Class
```
Solutions:
1. Check class balance in data
2. Adjust class weights in loss function
3. Verify label encoding is correct
4. Try different random seed
```

### General Issues

**Problem**: Can't Find Data Files
```
Solution: Check paths
ls testdata/
# Should show:
# train_postpartum_depression.csv
# test_postpartum_depression.csv
```

**Problem**: Import Errors
```
Solution: Install/update dependencies
pip install --upgrade transformers datasets scikit-learn pandas
```

---

## Advanced Topics

### Data Augmentation

Improve performance with synthetic data:

```python
# Text-based augmentation
- Paraphrase templates
- Synonym replacement
- Back-translation

# Tabular augmentation
- SMOTE for minority class
- Feature noise injection
```

### Ensemble Methods

Combine both approaches:

```python
# Weighted ensemble
final_pred = 0.6 * finetuning_pred + 0.4 * fewshot_pred

# Voting ensemble
final_pred = majority_vote([finetuning_pred, fewshot_pred, baseline_pred])
```

### Explainability

Understand model predictions:

1. **SHAP Values**: Feature importance
2. **Attention Visualization**: What model focuses on
3. **Example-based Explanations**: Similar training examples

---

## Best Practices

### 1. Start Small
- Test with `--max_samples 50` first
- Verify pipeline works end-to-end
- Then scale to full dataset

### 2. Experiment Tracking
- Keep log of all experiments
- Record hyperparameters
- Save all outputs

### 3. Validation
- Use cross-validation for small datasets
- Check for overfitting
- Compare with baseline (TF-IDF + Logistic Regression)

### 4. Error Analysis
- Examine misclassified samples
- Look for patterns
- Adjust templates or features

### 5. Clinical Validation
- Consult domain experts
- Validate on held-out data
- Consider deployment constraints

---

## Additional Resources

### Papers
- [TabLLM Paper](https://arxiv.org/pdf/2210.10723)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Few-Shot Learning Survey](https://arxiv.org/abs/2203.14713)

### Code Repositories
- [TabLLM GitHub](https://github.com/clinicalml/TabLLM)
- [T-Few GitHub](https://github.com/r-three/t-few)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com)
- [Hugging Face Docs](https://huggingface.co/docs)

---

## Citation

If you use this implementation, please cite the TabLLM paper:

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

---

## Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review the [TabLLM GitHub Issues](https://github.com/clinicalml/TabLLM/issues)
3. Consult the original paper and documentation

---

**Last Updated**: November 2025
**Version**: 1.0
**Dataset**: Postpartum Depression Classification
