"""
Compare Results from TabLLM Few-Shot and Fine-Tuning Approaches

This script compares the performance of both TabLLM approaches
and generates a comprehensive comparison report.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path, approach_type: str) -> Dict:
    """
    Load results from a results directory

    Args:
        results_dir: Path to results directory
        approach_type: 'fewshot' or 'finetuning'

    Returns:
        Dictionary containing metrics and predictions
    """
    logger.info(f"Loading {approach_type} results from {results_dir}")

    metrics_file = results_dir / f"{approach_type}_metrics.json"
    predictions_file = results_dir / f"{approach_type}_predictions.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)

    # Load predictions
    predictions_df = pd.read_csv(predictions_file)

    return {
        'metrics': metrics_data.get('metrics', metrics_data),
        'predictions': predictions_df,
        'config': metrics_data.get('config', {})
    }


def compare_metrics(fewshot_metrics: Dict, finetuning_metrics: Dict) -> pd.DataFrame:
    """
    Compare metrics between approaches

    Returns:
        DataFrame with comparison
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    comparison_data = []
    for metric in metrics_names:
        fewshot_val = fewshot_metrics.get(metric, 0)
        finetuning_val = finetuning_metrics.get(metric, 0)
        diff = finetuning_val - fewshot_val
        diff_pct = (diff / fewshot_val * 100) if fewshot_val > 0 else 0

        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Few-Shot': f"{fewshot_val:.4f} ({fewshot_val*100:.2f}%)",
            'Fine-Tuning': f"{finetuning_val:.4f} ({finetuning_val*100:.2f}%)",
            'Difference': f"{diff:+.4f}",
            'Difference (%)': f"{diff_pct:+.2f}%"
        })

    return pd.DataFrame(comparison_data)


def analyze_agreement(fewshot_preds: pd.DataFrame, finetuning_preds: pd.DataFrame) -> Dict:
    """
    Analyze agreement between approaches

    Returns:
        Dictionary with agreement statistics
    """
    # Ensure same length
    min_len = min(len(fewshot_preds), len(finetuning_preds))
    fewshot_preds = fewshot_preds.iloc[:min_len]
    finetuning_preds = finetuning_preds.iloc[:min_len]

    # Get predicted labels (convert to binary if needed)
    def get_binary_pred(df):
        if 'predicted_binary' in df.columns:
            return df['predicted_binary'].values
        elif 'predicted_label' in df.columns:
            preds = df['predicted_label'].values
            if preds.dtype == 'object':
                return np.array([1 if str(p).lower() == 'yes' else 0 for p in preds])
            return preds
        else:
            raise ValueError("No predicted_label column found")

    fewshot_binary = get_binary_pred(fewshot_preds)
    finetuning_binary = get_binary_pred(finetuning_preds)

    # Calculate agreement
    agreement = np.mean(fewshot_binary == finetuning_binary)

    # Both correct
    true_labels = fewshot_preds['true_label'].values
    if true_labels.dtype == 'object':
        true_binary = np.array([1 if str(t).lower() == 'yes' else 0 for t in true_labels])
    else:
        true_binary = true_labels

    both_correct = np.mean((fewshot_binary == true_binary) & (finetuning_binary == true_binary))

    # One correct
    fewshot_only = np.mean((fewshot_binary == true_binary) & (finetuning_binary != true_binary))
    finetuning_only = np.mean((finetuning_binary == true_binary) & (fewshot_binary != true_binary))

    # Both wrong
    both_wrong = np.mean((fewshot_binary != true_binary) & (finetuning_binary != true_binary))

    return {
        'agreement': agreement,
        'both_correct': both_correct,
        'fewshot_only_correct': fewshot_only,
        'finetuning_only_correct': finetuning_only,
        'both_wrong': both_wrong
    }


def create_comparison_plots(
    comparison_df: pd.DataFrame,
    agreement_stats: Dict,
    output_dir: Path
):
    """Create visualization plots"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Metrics comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = comparison_df['Metric'].values
    fewshot_vals = [float(x.split('(')[0]) for x in comparison_df['Few-Shot'].values]
    finetuning_vals = [float(x.split('(')[0]) for x in comparison_df['Fine-Tuning'].values]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, fewshot_vals, width, label='Few-Shot', color='skyblue')
    bars2 = ax.bar(x + width/2, finetuning_vals, width, label='Fine-Tuning', color='lightcoral')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('TabLLM Approach Comparison - Postpartum Depression Classification',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_dir / 'metrics_comparison.png'}")
    plt.close()

    # 2. Agreement pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    agreement_data = [
        agreement_stats['both_correct'] * 100,
        agreement_stats['fewshot_only_correct'] * 100,
        agreement_stats['finetuning_only_correct'] * 100,
        agreement_stats['both_wrong'] * 100
    ]

    labels = [
        'Both Correct',
        'Few-Shot Only',
        'Fine-Tuning Only',
        'Both Wrong'
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    wedges, texts, autotexts = ax.pie(
        agreement_data,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Prediction Agreement Analysis',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_dir / 'agreement_analysis.png'}")
    plt.close()


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    agreement_stats: Dict,
    fewshot_config: Dict,
    finetuning_config: Dict,
    output_dir: Path
):
    """Generate a comprehensive comparison report"""

    report_path = output_dir / 'comparison_report.md'

    with open(report_path, 'w') as f:
        f.write("# TabLLM Approach Comparison Report\n\n")
        f.write("## Postpartum Depression Classification\n\n")
        f.write("---\n\n")

        # Overview
        f.write("## 1. Overview\n\n")
        f.write("This report compares the performance of two TabLLM approaches:\n\n")
        f.write("1. **Few-Shot Learning**: In-context learning with LLM API\n")
        f.write("2. **Fine-Tuning**: Parameter-efficient fine-tuning of T5 model\n\n")
        f.write("---\n\n")

        # Configuration
        f.write("## 2. Configuration\n\n")

        f.write("### Few-Shot Configuration\n\n")
        if fewshot_config:
            f.write("```\n")
            for key, value in fewshot_config.items():
                f.write(f"{key}: {value}\n")
            f.write("```\n\n")
        else:
            f.write("Configuration not available\n\n")

        f.write("### Fine-Tuning Configuration\n\n")
        if finetuning_config:
            f.write("```\n")
            for key, value in finetuning_config.items():
                if key != 'classification_report':  # Skip large nested data
                    f.write(f"{key}: {value}\n")
            f.write("```\n\n")
        else:
            f.write("Configuration not available\n\n")

        f.write("---\n\n")

        # Metrics Comparison
        f.write("## 3. Performance Metrics Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")

        # Winner determination
        f.write("### Performance Summary\n\n")

        fewshot_wins = 0
        finetuning_wins = 0
        ties = 0

        for _, row in comparison_df.iterrows():
            diff = float(row['Difference'])
            if abs(diff) < 0.01:
                ties += 1
            elif diff > 0:
                finetuning_wins += 1
            else:
                fewshot_wins += 1

        f.write(f"- **Fine-Tuning Wins**: {finetuning_wins} metrics\n")
        f.write(f"- **Few-Shot Wins**: {fewshot_wins} metrics\n")
        f.write(f"- **Ties**: {ties} metrics\n\n")

        if finetuning_wins > fewshot_wins:
            f.write("**Overall Winner**: Fine-Tuning ✅\n\n")
        elif fewshot_wins > finetuning_wins:
            f.write("**Overall Winner**: Few-Shot ✅\n\n")
        else:
            f.write("**Overall Result**: Tie 🤝\n\n")

        f.write("---\n\n")

        # Agreement Analysis
        f.write("## 4. Agreement Analysis\n\n")
        f.write(f"- **Overall Agreement**: {agreement_stats['agreement']*100:.2f}%\n")
        f.write(f"- **Both Correct**: {agreement_stats['both_correct']*100:.2f}%\n")
        f.write(f"- **Few-Shot Only Correct**: {agreement_stats['fewshot_only_correct']*100:.2f}%\n")
        f.write(f"- **Fine-Tuning Only Correct**: {agreement_stats['finetuning_only_correct']*100:.2f}%\n")
        f.write(f"- **Both Wrong**: {agreement_stats['both_wrong']*100:.2f}%\n\n")

        f.write("### Interpretation\n\n")
        if agreement_stats['agreement'] > 0.85:
            f.write("The approaches show **high agreement** (>85%), suggesting they've learned similar patterns.\n\n")
        elif agreement_stats['agreement'] > 0.70:
            f.write("The approaches show **moderate agreement** (70-85%), with some complementary strengths.\n\n")
        else:
            f.write("The approaches show **low agreement** (<70%), suggesting different learning strategies.\n\n")

        f.write("---\n\n")

        # Recommendations
        f.write("## 5. Recommendations\n\n")

        # Get best performing approach
        avg_fewshot = np.mean([float(x.split('(')[0]) for x in comparison_df['Few-Shot'].values])
        avg_finetuning = np.mean([float(x.split('(')[0]) for x in comparison_df['Fine-Tuning'].values])

        f.write("### For Production Deployment\n\n")
        if avg_finetuning > avg_fewshot:
            f.write("✅ **Recommended: Fine-Tuning**\n\n")
            f.write("Reasons:\n")
            f.write("- Better overall performance\n")
            f.write("- No ongoing API costs\n")
            f.write("- Faster inference\n")
            f.write("- Works offline\n\n")
        else:
            f.write("✅ **Recommended: Few-Shot**\n\n")
            f.write("Reasons:\n")
            f.write("- Better overall performance\n")
            f.write("- No training required\n")
            f.write("- Easy to update examples\n")
            f.write("- Leverages powerful pre-trained models\n\n")

        f.write("### For Research/Experimentation\n\n")
        f.write("Consider using **Both Approaches**:\n")
        f.write("- Few-shot for rapid prototyping\n")
        f.write("- Fine-tuning for optimized performance\n")
        f.write("- Ensemble for maximum accuracy\n\n")

        f.write("---\n\n")

        # Ensemble Suggestion
        f.write("## 6. Ensemble Approach\n\n")
        f.write("Combining both approaches could yield better results:\n\n")
        f.write("### Simple Voting Ensemble\n")
        f.write("```python\n")
        f.write("# Take majority vote from both predictions\n")
        f.write("ensemble_pred = (fewshot_pred + finetuning_pred) >= 1\n")
        f.write("```\n\n")

        f.write("### Weighted Ensemble\n")
        weight_fewshot = avg_fewshot / (avg_fewshot + avg_finetuning)
        weight_finetuning = avg_finetuning / (avg_fewshot + avg_finetuning)
        f.write("```python\n")
        f.write(f"# Weight by performance ({weight_fewshot:.2f} / {weight_finetuning:.2f})\n")
        f.write(f"ensemble_score = {weight_fewshot:.2f} * fewshot_prob + {weight_finetuning:.2f} * finetuning_prob\n")
        f.write("```\n\n")

        f.write("**Potential Improvement**: +2-5% accuracy\n\n")

        f.write("---\n\n")

        # Clinical Implications
        f.write("## 7. Clinical Implications\n\n")
        f.write("### Screening vs Diagnosis\n\n")
        f.write("⚠️ **Important**: These models are for **screening**, not diagnosis.\n\n")
        f.write("### Use Cases\n\n")
        f.write("1. **Initial Screening**: Identify at-risk patients for follow-up\n")
        f.write("2. **Triage**: Prioritize patients for clinical assessment\n")
        f.write("3. **Monitoring**: Track changes over time\n\n")

        f.write("### Recommended Workflow\n\n")
        f.write("1. Patient completes questionnaire\n")
        f.write("2. Model provides risk score\n")
        f.write("3. High-risk patients → Clinical assessment\n")
        f.write("4. Clinical professional makes final diagnosis\n\n")

        f.write("---\n\n")

        # Next Steps
        f.write("## 8. Next Steps\n\n")
        f.write("1. **Error Analysis**: Examine misclassified samples\n")
        f.write("2. **Hyperparameter Tuning**: Optimize both approaches\n")
        f.write("3. **Data Augmentation**: Generate synthetic samples\n")
        f.write("4. **Ensemble Testing**: Evaluate combined predictions\n")
        f.write("5. **Clinical Validation**: Test with domain experts\n")
        f.write("6. **Deployment**: Integrate into clinical workflow\n\n")

        f.write("---\n\n")

        f.write("**Generated**: November 2025\n\n")
        f.write("**Dataset**: Postpartum Depression Classification\n\n")

    logger.info(f"Saved comparison report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare TabLLM Few-Shot and Fine-Tuning Results")
    parser.add_argument(
        "--fewshot_dir",
        type=str,
        required=True,
        help="Directory containing few-shot results"
    )
    parser.add_argument(
        "--finetuning_dir",
        type=str,
        required=True,
        help="Directory containing fine-tuning results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tabllm_comparison",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots (requires matplotlib)"
    )

    args = parser.parse_args()

    # Load results
    logger.info("Loading results...")
    fewshot_results = load_results(Path(args.fewshot_dir), 'fewshot')
    finetuning_results = load_results(Path(args.finetuning_dir), 'finetuning')

    # Compare metrics
    logger.info("Comparing metrics...")
    comparison_df = compare_metrics(
        fewshot_results['metrics'],
        finetuning_results['metrics']
    )

    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")

    # Analyze agreement
    logger.info("Analyzing prediction agreement...")
    agreement_stats = analyze_agreement(
        fewshot_results['predictions'],
        finetuning_results['predictions']
    )

    print("AGREEMENT ANALYSIS")
    print("="*80)
    for key, value in agreement_stats.items():
        print(f"{key.replace('_', ' ').title()}: {value*100:.2f}%")
    print("="*80 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison_df.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    logger.info(f"Saved metrics comparison: {output_dir / 'metrics_comparison.csv'}")

    # Save agreement stats
    with open(output_dir / 'agreement_stats.json', 'w') as f:
        json.dump(agreement_stats, f, indent=2)
    logger.info(f"Saved agreement stats: {output_dir / 'agreement_stats.json'}")

    # Create plots if requested
    if args.create_plots:
        try:
            logger.info("Creating visualization plots...")
            create_comparison_plots(comparison_df, agreement_stats, output_dir)
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            logger.info("Skipping plots. Install matplotlib and seaborn to enable.")

    # Generate report
    logger.info("Generating comparison report...")
    generate_comparison_report(
        comparison_df,
        agreement_stats,
        fewshot_results['config'],
        finetuning_results['config'],
        output_dir
    )

    print(f"\n✅ Comparison complete! Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - metrics_comparison.csv")
    print(f"  - agreement_stats.json")
    print(f"  - comparison_report.md")
    if args.create_plots:
        print(f"  - metrics_comparison.png")
        print(f"  - agreement_analysis.png")


if __name__ == '__main__':
    main()
