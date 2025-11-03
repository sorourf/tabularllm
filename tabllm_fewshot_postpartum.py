"""
TabLLM Few-Shot Approach for Postpartum Depression Classification

This script implements the few-shot learning approach from the TabLLM paper.
It uses LLM APIs (OpenAI GPT-4 or similar) to perform classification with few examples.

Key features:
1. Converts tabular data to text using TabLLM templates
2. Creates few-shot prompts with k training examples
3. Uses LLM API for inference
4. Evaluates performance on test set

Paper: https://arxiv.org/pdf/2210.10723
"""

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

from helper.note_template import NoteTemplate
from helper.external_datasets_variables import (
    template_postpartum_depression_list,
    template_config_postpartum_depression_list
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_postpartum_depression_data(data_dir):
    """Load and preprocess postpartum depression dataset"""
    logger.info("Loading postpartum depression dataset...")

    # Load train and test data
    train_df = pd.read_csv(data_dir / 'train_postpartum_depression.csv')
    test_df = pd.read_csv(data_dir / 'test_postpartum_depression.csv')

    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Column mapping: rename columns to match template variable names
    column_mapping = {
        'timestamp': 'timestamp',
        'age': 'age',
        'feeling sad or tearful': 'feeling_sad_or_tearful',
        'irritable towards baby & partner': 'irritable_towards_baby_partner',
        'trouble sleeping at night': 'trouble_sleeping_at_night',
        'problems concentrating or making decision': 'problems_concentrating_or_making_decision',
        'overeating or loss of appetite': 'overeating_or_loss_of_appetite',
        'feeling anxious': 'label',
        'feeling of guilt': 'feeling_of_guilt',
        'problems of bonding with baby': 'problems_of_bonding_with_baby',
        'suicide attempt': 'suicide_attempt'
    }

    train_df = train_df.rename(columns=column_mapping)
    test_df = test_df.rename(columns=column_mapping)

    # Drop timestamp
    train_df = train_df.drop(columns=['timestamp'])
    test_df = test_df.drop(columns=['timestamp'])

    # Convert label to binary
    label_mapping = {'Yes': 1, 'No': 0}
    train_df['label'] = train_df['label'].map(label_mapping).fillna(0).astype(int)
    test_df['label'] = test_df['label'].map(label_mapping).fillna(0).astype(int)

    logger.info(f"Training label distribution:\n{train_df['label'].value_counts()}")
    logger.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

    return train_df, test_df


def serialize_to_text(df, template, template_config):
    """Convert tabular data to text using TabLLM templates"""
    logger.info("Serializing tabular data to text sequences...")

    note_template = NoteTemplate(template, **template_config)

    texts = []
    for idx, row in df.iterrows():
        try:
            text = note_template.substitute(row)
            text = clean_note(text)
            texts.append(text)
        except Exception as e:
            logger.error(f"Error serializing row {idx}: {e}")
            texts.append("")

    logger.info(f"Serialized {len(texts)} samples")
    if len(texts) > 0 and texts[0]:
        logger.info(f"Example serialized text:\n{texts[0]}\n")

    return texts


def clean_note(note):
    """Clean generated note text"""
    import re
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub("\n\n\n+", "\n\n", note)
    note = re.sub(r"^[ \t]+", "", note)
    note = re.sub(r"\n[ \t]+", "\n", note)
    note = re.sub(r"[ \t]$", "", note)
    note = re.sub(r"[ \t]\n", "\n", note)
    return note.strip()


def create_fewshot_prompt(
    train_texts: List[str],
    train_labels: List[int],
    test_text: str,
    num_shots: int = 4,
    seed: int = 42
) -> str:
    """
    Create a few-shot prompt for classification

    Args:
        train_texts: List of training example texts
        train_labels: List of training labels
        test_text: Text to classify
        num_shots: Number of examples to include in prompt
        seed: Random seed for example selection

    Returns:
        Formatted prompt string
    """
    # Select balanced examples (equal from each class if possible)
    random.seed(seed)

    # Separate by class
    class_0_indices = [i for i, label in enumerate(train_labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(train_labels) if label == 1]

    # Select examples
    shots_per_class = num_shots // 2
    selected_indices = []

    if len(class_0_indices) >= shots_per_class:
        selected_indices.extend(random.sample(class_0_indices, shots_per_class))
    else:
        selected_indices.extend(class_0_indices)

    if len(class_1_indices) >= shots_per_class:
        selected_indices.extend(random.sample(class_1_indices, shots_per_class))
    else:
        selected_indices.extend(class_1_indices)

    # If we need more examples, add randomly
    while len(selected_indices) < num_shots and len(selected_indices) < len(train_labels):
        remaining = list(set(range(len(train_labels))) - set(selected_indices))
        if remaining:
            selected_indices.append(random.choice(remaining))
        else:
            break

    random.shuffle(selected_indices)

    # Build prompt
    prompt = """You are a medical AI assistant specialized in postpartum depression classification.
Based on maternal health indicators, classify whether the patient is experiencing anxiety (a predictor of postpartum depression).

For each patient profile, respond with ONLY "Yes" or "No" indicating whether they are feeling anxious.

Here are some examples:

"""

    # Add few-shot examples
    for idx in selected_indices:
        example_text = train_texts[idx]
        example_label = "Yes" if train_labels[idx] == 1 else "No"
        prompt += f"Patient Profile:\n{example_text}\n\nFeeling Anxious: {example_label}\n\n---\n\n"

    # Add test example
    prompt += f"Now classify this patient:\n\nPatient Profile:\n{test_text}\n\nFeeling Anxious:"

    return prompt


def query_llm_api(prompt: str, api_type: str = "openai", model: str = "gpt-4", temperature: float = 0.0) -> str:
    """
    Query LLM API for prediction

    Args:
        prompt: Input prompt
        api_type: Type of API ("openai", "anthropic", "local")
        model: Model name
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    if api_type == "openai":
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=10
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error querying OpenAI API: {e}")
            raise

    elif api_type == "anthropic":
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model=model,
                max_tokens=10,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text.strip()

        except ImportError:
            logger.error("Anthropic library not installed. Run: pip install anthropic")
            raise
        except Exception as e:
            logger.error(f"Error querying Anthropic API: {e}")
            raise

    elif api_type == "local":
        # Placeholder for local model inference
        # You can implement this with Hugging Face transformers
        logger.warning("Local inference not implemented. Returning random prediction.")
        return random.choice(["Yes", "No"])

    else:
        raise ValueError(f"Unknown API type: {api_type}")


def parse_prediction(response: str) -> int:
    """
    Parse LLM response to binary prediction

    Args:
        response: LLM response text

    Returns:
        Binary prediction (0 or 1)
    """
    response = response.strip().lower()

    # Check for common positive responses
    if any(word in response for word in ["yes", "anxious", "positive", "1"]):
        return 1
    elif any(word in response for word in ["no", "not anxious", "negative", "0"]):
        return 0
    else:
        # Default to 0 if unclear
        logger.warning(f"Unclear response: '{response}'. Defaulting to 0.")
        return 0


def run_fewshot_evaluation(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    num_shots: int = 4,
    api_type: str = "openai",
    model: str = "gpt-4",
    seed: int = 42,
    batch_delay: float = 1.0,
    max_samples: int = None
):
    """
    Run few-shot evaluation on test set

    Args:
        train_texts: Training example texts
        train_labels: Training labels
        test_texts: Test example texts
        test_labels: Test labels
        num_shots: Number of shots for few-shot learning
        api_type: API type ("openai", "anthropic", "local")
        model: Model name
        seed: Random seed
        batch_delay: Delay between API calls (seconds)
        max_samples: Maximum number of test samples to evaluate (None for all)

    Returns:
        Dictionary of predictions and metadata
    """
    logger.info(f"Starting few-shot evaluation with {num_shots} shots")
    logger.info(f"API: {api_type}, Model: {model}")

    predictions = []
    responses = []

    # Limit test samples if specified
    if max_samples is not None and max_samples < len(test_texts):
        logger.info(f"Limiting evaluation to {max_samples} samples")
        test_texts = test_texts[:max_samples]
        test_labels = test_labels[:max_samples]

    total_samples = len(test_texts)

    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        logger.info(f"Processing sample {i+1}/{total_samples}")

        try:
            # Create prompt
            prompt = create_fewshot_prompt(train_texts, train_labels, text, num_shots, seed)

            # Query API
            response = query_llm_api(prompt, api_type, model)

            # Parse prediction
            prediction = parse_prediction(response)

            predictions.append(prediction)
            responses.append(response)

            logger.info(f"Sample {i+1}: True={true_label}, Pred={prediction}, Response='{response}'")

            # Delay to avoid rate limits
            if i < total_samples - 1:
                time.sleep(batch_delay)

        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            # Use default prediction
            predictions.append(0)
            responses.append("ERROR")

    return {
        'predictions': predictions,
        'responses': responses,
        'true_labels': test_labels
    }


def evaluate_predictions(predictions: List[int], true_labels: List[int]) -> Dict:
    """Calculate evaluation metrics"""

    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='binary', zero_division=0),
        'recall': recall_score(true_labels, predictions, average='binary', zero_division=0),
        'f1_score': f1_score(true_labels, predictions, average='binary', zero_division=0),
    }

    # AUC-ROC (use predictions as probabilities)
    try:
        metrics['auc_roc'] = roc_auc_score(true_labels, predictions)
    except:
        metrics['auc_roc'] = 0.5  # Default if can't calculate

    cm = confusion_matrix(true_labels, predictions)

    logger.info("\n" + "="*80)
    logger.info("TABLLM FEW-SHOT EVALUATION RESULTS - Postpartum Depression")
    logger.info("="*80)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    logger.info(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
    logger.info("\nConfusion Matrix:")
    logger.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    logger.info("="*80 + "\n")

    logger.info("Detailed Classification Report:")
    logger.info("\n" + classification_report(true_labels, predictions, target_names=['No Anxiety', 'Anxiety']))

    metrics['confusion_matrix'] = cm.tolist()
    metrics['classification_report'] = classification_report(true_labels, predictions, output_dict=True)

    return metrics


def save_results(results: Dict, metrics: Dict, output_dir: Path, config: Dict):
    """Save evaluation results to files"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / 'fewshot_metrics.json', 'w') as f:
        json.dump({
            'config': config,
            'metrics': metrics
        }, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': results['true_labels'],
        'predicted_label': results['predictions'],
        'llm_response': results['responses']
    })
    predictions_df.to_csv(output_dir / 'fewshot_predictions.csv', index=False)

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TabLLM Few-Shot Evaluation for Postpartum Depression")
    parser.add_argument("--data_dir", type=str, default="testdata", help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="tabllm_fewshot_results", help="Directory to save results")
    parser.add_argument("--num_shots", type=int, default=4, help="Number of few-shot examples")
    parser.add_argument("--api_type", type=str, default="openai", choices=["openai", "anthropic", "local"], help="LLM API type")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum test samples to evaluate")

    args = parser.parse_args()

    config = vars(args)

    # Check for API key
    if args.api_type == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    elif args.api_type == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set!")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Load data
    data_dir = Path(args.data_dir)
    train_df, test_df = load_postpartum_depression_data(data_dir)

    # Serialize to text
    train_texts = serialize_to_text(
        train_df.drop(columns=['label']),
        template_postpartum_depression_list,
        template_config_postpartum_depression_list
    )

    test_texts = serialize_to_text(
        test_df.drop(columns=['label']),
        template_postpartum_depression_list,
        template_config_postpartum_depression_list
    )

    # Run few-shot evaluation
    results = run_fewshot_evaluation(
        train_texts=train_texts,
        train_labels=train_df['label'].tolist(),
        test_texts=test_texts,
        test_labels=test_df['label'].tolist(),
        num_shots=args.num_shots,
        api_type=args.api_type,
        model=args.model,
        seed=args.seed,
        batch_delay=args.batch_delay,
        max_samples=args.max_samples
    )

    # Evaluate
    metrics = evaluate_predictions(results['predictions'], results['true_labels'])

    # Save results
    save_results(results, metrics, Path(args.output_dir), config)

    logger.info("Few-shot evaluation complete!")


if __name__ == '__main__':
    main()
