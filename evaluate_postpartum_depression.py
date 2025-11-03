"""
TabLLM Evaluation Script for Postpartum Depression Dataset

This script:
1. Converts tabular data to text using TabLLM's template-based serialization
2. Uses TF-IDF vectorization and Logistic Regression as a lightweight baseline
3. Evaluates performance with comprehensive metrics
4. Saves predictions and results

This provides a practical evaluation of TabLLM's text serialization approach
without requiring the full T0/T-Few LLM training pipeline.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from datasets import Dataset
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
    # Remove spaces and special characters
    column_mapping = {
        'timestamp': 'timestamp',
        'age': 'age',
        'feeling sad or tearful': 'feeling_sad_or_tearful',
        'irritable towards baby & partner': 'irritable_towards_baby_partner',
        'trouble sleeping at night': 'trouble_sleeping_at_night',
        'problems concentrating or making decision': 'problems_concentrating_or_making_decision',
        'overeating or loss of appetite': 'overeating_or_loss_of_appetite',
        'feeling anxious': 'label',  # This is the target variable
        'feeling of guilt': 'feeling_of_guilt',
        'problems of bonding with baby': 'problems_of_bonding_with_baby',
        'suicide attempt': 'suicide_attempt'
    }

    train_df = train_df.rename(columns=column_mapping)
    test_df = test_df.rename(columns=column_mapping)

    # Drop timestamp column as it's not useful for prediction
    train_df = train_df.drop(columns=['timestamp'])
    test_df = test_df.drop(columns=['timestamp'])

    # Convert label to binary (Yes=1, No=0)
    # Handle different possible values
    label_mapping = {'Yes': 1, 'No': 0}

    # For training data
    if train_df['label'].dtype == 'object':
        train_df['label'] = train_df['label'].map(label_mapping)

    # For test data
    if test_df['label'].dtype == 'object':
        test_df['label'] = test_df['label'].map(label_mapping)

    # Check for any unmapped values
    if train_df['label'].isna().any():
        logger.warning(f"Training data has {train_df['label'].isna().sum()} unmapped label values")
        logger.warning(f"Unique unmapped values: {train_df[train_df['label'].isna()]['label'].unique()}")
        # Fill NaN with 0 as conservative approach
        train_df['label'] = train_df['label'].fillna(0)

    if test_df['label'].isna().any():
        logger.warning(f"Test data has {test_df['label'].isna().sum()} unmapped label values")
        logger.warning(f"Unique unmapped values: {test_df[test_df['label'].isna()]['label'].unique()}")
        test_df['label'] = test_df['label'].fillna(0)

    # Convert to int
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    # Check label distribution
    logger.info(f"Training label distribution:\n{train_df['label'].value_counts()}")
    logger.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

    return train_df, test_df


def serialize_to_text(df, template, template_config):
    """
    Convert tabular data to text using TabLLM templates

    This is the core TabLLM approach: converting structured data to natural language
    """
    logger.info("Serializing tabular data to text sequences...")

    note_template = NoteTemplate(template, **template_config)

    # Generate text for each row
    texts = []
    for idx, row in df.iterrows():
        try:
            text = note_template.substitute(row)
            # Clean the text
            text = clean_note(text)
            texts.append(text)
        except Exception as e:
            logger.error(f"Error serializing row {idx}: {e}")
            logger.error(f"Row data: {row.to_dict()}")
            texts.append("")  # Fallback to empty string

    logger.info(f"Serialized {len(texts)} samples")
    if len(texts) > 0 and texts[0]:
        logger.info(f"Example serialized text:\n{texts[0]}\n")

    return texts


def clean_note(note):
    """Clean generated note text"""
    import re
    # Remove multiple whitespaces
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub("\n\n\n+", "\n\n", note)
    # Remove leading/trailing whitespaces
    note = re.sub(r"^[ \t]+", "", note)
    note = re.sub(r"\n[ \t]+", "\n", note)
    note = re.sub(r"[ \t]$", "", note)
    note = re.sub(r"[ \t]\n", "\n", note)
    return note.strip()


def train_and_evaluate_classifier(train_texts, train_labels, test_texts, test_labels, output_dir):
    """
    Train classifier on text-serialized data and evaluate

    Uses TF-IDF + Logistic Regression as a simple but effective baseline
    for evaluating the quality of TabLLM's text serialization
    """
    logger.info("Training classifier on serialized text data...")

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    logger.info(f"TF-IDF feature shape - Train: {X_train.shape}, Test: {X_test.shape}")

    # Train logistic regression classifier
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0
    )

    classifier.fit(X_train, train_labels)
    logger.info("Classifier trained successfully")

    # Make predictions
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(test_labels, y_pred),
        'precision': precision_score(test_labels, y_pred, average='binary', zero_division=0),
        'recall': recall_score(test_labels, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(test_labels, y_pred, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(test_labels, y_pred_proba)
    }

    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("TABLLM EVALUATION RESULTS - Postpartum Depression Classification")
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

    # Detailed classification report
    logger.info("Detailed Classification Report:")
    logger.info("\n" + classification_report(test_labels, y_pred, target_names=['No Anxiety', 'Anxiety']))

    # Save results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(test_labels, y_pred, output_dict=True)
    }

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_proba
    })

    # Save to files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'tabllm_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    predictions_df.to_csv(output_dir / 'tabllm_predictions.csv', index=False)

    # Save serialized texts for inspection
    with open(output_dir / 'serialized_samples.txt', 'w') as f:
        f.write("Sample Serialized Texts (first 10 train, first 10 test):\n\n")
        f.write("="*80 + "\n")
        f.write("TRAINING SAMPLES:\n")
        f.write("="*80 + "\n")
        for i, text in enumerate(train_texts[:10]):
            f.write(f"\nSample {i+1} (Label: {train_labels.iloc[i]}):\n")
            f.write(text + "\n")
            f.write("-"*80 + "\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TEST SAMPLES:\n")
        f.write("="*80 + "\n")
        for i, text in enumerate(test_texts[:10]):
            f.write(f"\nSample {i+1} (Label: {test_labels.iloc[i]}, Predicted: {y_pred[i]}):\n")
            f.write(text + "\n")
            f.write("-"*80 + "\n")

    logger.info(f"Results saved to {output_dir}")

    return metrics, predictions_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate TabLLM on Postpartum Depression Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="testdata",
        help="Directory containing the dataset files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tabllm_postpartum_results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Load data
    data_dir = Path(args.data_dir)
    train_df, test_df = load_postpartum_depression_data(data_dir)

    # Serialize to text using TabLLM templates
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

    # Train and evaluate
    metrics, predictions = train_and_evaluate_classifier(
        train_texts,
        train_df['label'],
        test_texts,
        test_df['label'],
        args.output_dir
    )

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
