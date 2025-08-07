"""
Visualization utilities for AWS API Label Propagation.

This module contains functions for visualizing the results of label propagation,
including confusion matrices, label distributions, and analysis summaries.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], 
                         title: str = "Label Propagation - Confusion Matrix",
                         figsize: tuple = (8, 6)) -> None:
    """
    Plot confusion matrix for label propagation results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size tuple
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['none', 'sink', 'source'], 
               yticklabels=['none', 'sink', 'source'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_label_distribution(label_counts: Dict[str, Dict[str, int]], 
                           title: str = "Label Distribution by Service",
                           figsize: tuple = (16, 8),
                           total_methods: Dict[str, int] = None,
                           manual_labels: Dict[tuple, str] = None) -> None:
    """
    Plot label distribution across services with coverage percentages.
    
    Args:
        label_counts: Dictionary of service -> label -> count (from predictions)
        total_methods: Dictionary of service -> total method count
        manual_labels: Dictionary of (service, method) -> label (original labels)
    """
    services = list(label_counts.keys())
    labels = ['none', 'sink', 'source']
    
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(services)) * 1.75
    width = 0.25
    
    for i, label in enumerate(labels):
        counts = [label_counts[service].get(label, 0) for service in services]
        ax.bar(x + i * width, counts, width, label=label)
    
    # Add coverage percentages above bars
    if total_methods:
        for i, service in enumerate(services):
            total = total_methods.get(service, 0)
            
            # Count propagated predictions
            propagated_count = sum(label_counts[service].values())
            
            # Count original manual labels for this service
            manual_count = 0
            if manual_labels:
                manual_count = sum(1 for (label_service, method), label in manual_labels.items() 
                                 if label_service == service)
            
            # Total labeled = propagated + manual (avoiding double counting)
            # We assume propagated predictions don't include manual labels
            total_labeled = propagated_count + manual_count
            coverage = (total_labeled / total * 100) if total > 0 else 0
            
            # Find the highest bar for this service
            max_height = max([label_counts[service].get(label, 0) for label in labels])
            
            # Add coverage text
            ax.annotate(f'{coverage:.1f}%', 
                       xy=(x[i] + width, max_height), 
                       xytext=(5, 5), textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Services')
    ax.set_ylabel('Number of Methods')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(services, rotation=90, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(predictions: Dict[str, Dict[str, Any]], 
                                title: str = "Confidence Distribution",
                                figsize: tuple = (16, 6)) -> None:
    """
    Plot confidence score distribution for predictions.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        title: Plot title
        figsize: Figure size tuple
    """
    all_confidences = []
    service_confidences = {}
    
    for service, methods in predictions.items():
        confidences = [data['confidence'] for data in methods.values() if isinstance(data, dict)]
        service_confidences[service] = confidences
        all_confidences.extend(confidences)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Overall confidence distribution
    ax1.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Confidence Distribution')
    ax1.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_confidences):.3f}')
    ax1.legend()
    
    # Confidence by service (box plot)
    services = list(service_confidences.keys())
    confidence_data = [service_confidences[service] for service in services]
    
    ax2.boxplot(confidence_data, labels=services)
    ax2.set_xlabel('Service')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence by Service')
    ax2.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()


def plot_similarity_vs_confidence(predictions: Dict[str, Dict[str, Any]],
                                 title: str = "Similarity vs Confidence",
                                 figsize: tuple = (10, 6)) -> None:
    """
    Plot similarity vs confidence scatter plot.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        title: Plot title
        figsize: Figure size tuple
    """
    similarities = []
    confidences = []
    labels = []
    
    for service, methods in predictions.items():
        for method, data in methods.items():
            if isinstance(data, dict) and 'similarity' in data and 'confidence' in data:
                similarities.append(data['similarity'])
                confidences.append(data['confidence'])
                labels.append(data['label'])
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot with different colors for each label
    label_colors = {'source': 'red', 'sink': 'blue', 'none': 'gray'}
    
    for label in set(labels):
        mask = [l == label for l in labels]
        sim_subset = [s for s, m in zip(similarities, mask) if m]
        conf_subset = [c for c, m in zip(confidences, mask) if m]
        
        plt.scatter(sim_subset, conf_subset, 
                   c=label_colors.get(label, 'black'), 
                   label=label, alpha=0.6)
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Confidence Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cross_service_comparison(comparison_results: Dict, figsize: tuple = (14, 8)) -> None:
    """
    Create bar charts comparing group-based and all-to-all cross-service predictions.
    
    Args:
        comparison_results: Results from compare_cross_service_predictions
        figsize: Figure size tuple
    """
    if not comparison_results:
        print("❌ No comparison data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Cross-Service Prediction Comparison', fontsize=16, fontweight='bold')
    
    # 1. Agreement rates by service (only for services with common predictions)
    ax1 = axes[0, 0]
    agreement_stats = comparison_results.get('agreement_stats', {})
    
    if agreement_stats:
        agreement_services = list(agreement_stats.keys())
        agreement_rates = [stats['agreement_rate'] for stats in agreement_stats.values()]
        
        colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.6 else 'red' 
                  for rate in agreement_rates]
        
        bars = ax1.bar(range(len(agreement_services)), agreement_rates, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(agreement_services)))
        ax1.set_xticklabels(agreement_services, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Agreement Rate')
        ax1.set_title('Agreement Rates by Service')
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (≥0.8)')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.6)')
        ax1.set_ylim(0, 1.1)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'No common predictions found', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Agreement Rates by Service')
    
    # 2. Coverage comparison (for all services)
    ax2 = axes[0, 1]
    coverage_stats = comparison_results.get('coverage_comparison', {})
    
    if coverage_stats:
        coverage_services = list(coverage_stats.keys())[:20]  # Limit to 20 services for readability
        coverage_data = {s: coverage_stats[s] for s in coverage_services}
        
        group_coverage = [coverage_data[s]['group_predictions'] for s in coverage_services]
        all_to_all_coverage = [coverage_data[s]['all_to_all_predictions'] for s in coverage_services]
        
        x = np.arange(len(coverage_services))
        width = 0.35
        
        ax2.bar(x - width/2, group_coverage, width, label='Group-based', color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, all_to_all_coverage, width, label='All-to-all', color='coral', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(coverage_services, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Number of Predictions')
        ax2.set_title(f'Prediction Coverage (Top {len(coverage_services)} Services)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No coverage data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Prediction Coverage by Method')
    
    # 3. Label distribution in disagreements
    ax3 = axes[1, 0]
    disagreement_details = comparison_results.get('disagreement_details', {})
    
    label_disagreements = defaultdict(lambda: defaultdict(int))
    total_disagreements = 0
    
    for service, disagreements in disagreement_details.items():
        if isinstance(disagreements, list):
            for disagreement in disagreements:
                if isinstance(disagreement, dict) and 'group_prediction' in disagreement and 'all_to_all_prediction' in disagreement:
                    group_label = disagreement['group_prediction']
                    all_label = disagreement['all_to_all_prediction']
                    label_disagreements[group_label][all_label] += 1
                    total_disagreements += 1
    
    if total_disagreements > 0:
        labels = ['none', 'sink', 'source']
        matrix = np.zeros((len(labels), len(labels)))
        
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                matrix[i, j] = label_disagreements[label1][label2]
        
        im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax3.set_xticks(np.arange(len(labels)))
        ax3.set_yticks(np.arange(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_yticklabels(labels)
        ax3.set_xlabel('All-to-all Prediction')
        ax3.set_ylabel('Group-based Prediction')
        ax3.set_title(f'Label Disagreement Matrix ({total_disagreements} total)')
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i, j] > 0:
                    text = ax3.text(j, i, int(matrix[i, j]),
                                   ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, 'No disagreements found\n(Perfect agreement!)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Label Disagreement Matrix')
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = comparison_results.get('summary', {})
    
    # Calculate agreement quality stats only if we have agreement data
    agreement_quality_text = ""
    if agreement_stats:
        agreement_rates_list = [stats['agreement_rate'] for stats in agreement_stats.values()]
        agreement_quality_text = f"""
    Agreement Quality:
    • Excellent (≥80%): {sum(1 for rate in agreement_rates_list if rate >= 0.8)} services
    • Fair (60-80%): {sum(1 for rate in agreement_rates_list if 0.6 <= rate < 0.8)} services
    • Poor (<60%): {sum(1 for rate in agreement_rates_list if rate < 0.6)} services"""
    
    summary_text = f"""
    Overall Agreement: {summary.get('overall_agreement_rate', 0):.1%}
    
    Total Common Methods: {summary.get('total_common_predictions', 0)}
    Total Agreements: {summary.get('total_agreements', 0)}
    Total Disagreements: {summary.get('total_disagreements', 0)}
    
    Group-based Total: {summary.get('group_total_predictions', 0)}
    All-to-all Total: {summary.get('all_to_all_total_predictions', 0)}
    {agreement_quality_text}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.show()


def create_propagation_dashboard(predictions: Dict[str, Dict[str, Any]],
                                total_methods: Dict[str, int] = None,
                                manual_labels: Dict[str, str] = None,
                                title_prefix: str = "") -> None:
    """
    Create a comprehensive dashboard for a specific propagation type.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        total_methods: Dictionary of service -> total method count
        manual_labels: Dictionary of (service, method) -> label
        title_prefix: Prefix for the dashboard title (e.g., "Within-Service", "Cross-Service")
    """
    if not predictions:
        print(f"❌ No predictions available for {title_prefix} dashboard")
        return
    
    print(f"🎨 Creating {title_prefix} Visualization Dashboard")
    print("=" * 50)
    
    # 1. Confidence distribution
    plot_confidence_distribution(predictions, f"{title_prefix} - Confidence Distribution")
    
    # 2. Similarity vs Confidence scatter plot
    plot_similarity_vs_confidence(predictions, f"{title_prefix} - Similarity vs Confidence")
    
    # 3. Label distribution
    label_counts = {}
    for service, methods in predictions.items():
        service_counts = defaultdict(int)
        for method, data in methods.items():
            if isinstance(data, dict):
                service_counts[data['label']] += 1
        label_counts[service] = dict(service_counts)
    
    if label_counts:
        plot_label_distribution(label_counts, f"{title_prefix} - Label Distribution", 
                              total_methods=total_methods, 
                              manual_labels=manual_labels)
    
    print(f"✅ {title_prefix} dashboard complete!")


def print_evaluation_summary(evaluation_results: Dict[str, Any], 
                            best_k: int,
                            total_labeled: int,
                            total_predictions: int) -> None:
    """
    Print a formatted evaluation summary.
    
    Args:
        evaluation_results: Results from model evaluation
        best_k: Optimal k value found
        total_labeled: Total number of labeled methods
        total_predictions: Total number of predictions made
    """
    print("🎯 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"📊 Total labeled methods: {total_labeled}")
    print(f"🔍 Optimal k value: {best_k}")
    print(f"📈 Total predictions made: {total_predictions}")
    
    if evaluation_results:
        print(f"📊 Overall accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"📊 Macro F1-score: {evaluation_results['macro avg']['f1-score']:.3f}")
        print(f"📊 Weighted F1-score: {evaluation_results['weighted avg']['f1-score']:.3f}")
        
        print("\n📋 Per-class performance:")
        for label in ['none', 'sink', 'source']:
            if label in evaluation_results:
                metrics = evaluation_results[label]
                print(f"  {label:>6}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                      f"Support={metrics['support']}")


def save_visualization_summary(predictions: Dict[str, Dict[str, Any]], 
                              output_file: Path,
                              metadata: Dict[str, Any] = None) -> None:
    """
    Save visualization data summary to JSON file.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        output_file: Path to output file
        metadata: Optional metadata to include
    """
    summary = {
        'metadata': metadata or {},
        'visualization_data': {
            'total_predictions': sum(len(pred) for pred in predictions.values()),
            'services_analyzed': list(predictions.keys()),
            'statistics_by_service': {}
        }
    }
    
    for service, service_predictions in predictions.items():
        if not service_predictions:
            continue
            
        label_counts = defaultdict(int)
        confidences = []
        similarities = []
        
        for method, pred_data in service_predictions.items():
            if isinstance(pred_data, dict):
                label_counts[pred_data['label']] += 1
                confidences.append(pred_data['confidence'])
                if 'similarity' in pred_data:
                    similarities.append(pred_data['similarity'])
        
        summary['visualization_data']['statistics_by_service'][service] = {
            'total_predictions': len(service_predictions),
            'label_distribution': dict(label_counts),
            'confidence_stats': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'std': float(np.std(confidences)) if confidences else 0,
                'min': float(np.min(confidences)) if confidences else 0,
                'max': float(np.max(confidences)) if confidences else 0
            },
            'similarity_stats': {
                'mean': float(np.mean(similarities)) if similarities else 0,
                'std': float(np.std(similarities)) if similarities else 0,
                'min': float(np.min(similarities)) if similarities else 0,
                'max': float(np.max(similarities)) if similarities else 0
            } if similarities else None
        }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Visualization summary saved to: {output_file}")