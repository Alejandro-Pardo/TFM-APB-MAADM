"""
Visualization utilities for AWS API Label Propagation.

This module contains functions for visualizing the results of label propagation,
including confusion matrices, label distributions, and analysis summaries.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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
                           figsize: tuple = (12, 8)) -> None:
    """
    Plot label distribution across services.
    
    Args:
        label_counts: Dictionary of service -> label -> count
        title: Plot title
        figsize: Figure size tuple
    """
    services = list(label_counts.keys())
    labels = ['none', 'sink', 'source']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(services))
    width = 0.25
    
    for i, label in enumerate(labels):
        counts = [label_counts[service].get(label, 0) for service in services]
        ax.bar(x + i * width, counts, width, label=label)
    
    ax.set_xlabel('Services')
    ax.set_ylabel('Number of Methods')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(services, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(predictions: Dict[str, Dict[str, Any]], 
                                title: str = "Confidence Distribution",
                                figsize: tuple = (10, 6)) -> None:
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
    ax2.tick_params(axis='x', rotation=45)
    
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


def plot_threshold_analysis(results_by_threshold: Dict[float, Dict[str, Any]],
                           title: str = "Threshold Analysis",
                           figsize: tuple = (10, 6)) -> None:
    """
    Plot threshold analysis showing predictions count vs threshold.
    
    Args:
        results_by_threshold: Dictionary of threshold -> results
        title: Plot title
        figsize: Figure size tuple
    """
    thresholds = sorted(results_by_threshold.keys())
    total_counts = [results_by_threshold[t]['total_count'] for t in thresholds]
    
    plt.figure(figsize=figsize)
    plt.plot(thresholds, total_counts, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Number of Predictions')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (threshold, count) in enumerate(zip(thresholds, total_counts)):
        plt.annotate(f'{count}', (threshold, count), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()


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


def print_service_predictions_summary(predictions: Dict[str, Dict[str, Any]],
                                     service_name: str = None) -> None:
    """
    Print a formatted summary of predictions for services.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        service_name: Optional specific service to summarize
    """
    if service_name:
        services_to_show = [service_name] if service_name in predictions else []
        title = f"📊 PREDICTIONS SUMMARY - {service_name.upper()}"
    else:
        services_to_show = list(predictions.keys())
        title = "📊 PREDICTIONS SUMMARY - ALL SERVICES"
    
    print(title)
    print("=" * 50)
    
    for service in services_to_show:
        service_predictions = predictions[service]
        
        if not service_predictions:
            print(f"{service.upper()}: No predictions")
            continue
        
        # Calculate statistics
        label_counts = defaultdict(int)
        confidences = []
        similarities = []
        
        for method, pred_data in service_predictions.items():
            if isinstance(pred_data, dict):
                label_counts[pred_data['label']] += 1
                confidences.append(pred_data['confidence'])
                if 'similarity' in pred_data:
                    similarities.append(pred_data['similarity'])
        
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_similarity = np.mean(similarities) if similarities else 0
        
        print(f"{service.upper()}:")
        print(f"  📊 {len(service_predictions)} predictions made")
        print(f"  🏷️  Labels: {dict(label_counts)}")
        print(f"  📈 Avg confidence: {avg_confidence:.3f}")
        if similarities:
            print(f"  📈 Avg similarity: {avg_similarity:.3f}")
        
        # Show high-confidence predictions
        high_conf_methods = [method for method, data in service_predictions.items() 
                           if isinstance(data, dict) and data['confidence'] > 0.8]
        if high_conf_methods:
            print(f"  ✅ High-confidence ({len(high_conf_methods)} methods): "
                  f"{', '.join(high_conf_methods[:5])}"
                  f"{'...' if len(high_conf_methods) > 5 else ''}")
        
        print()


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


def create_results_dashboard(predictions: Dict[str, Dict[str, Any]],
                           evaluation_results: Dict[str, Any] = None,
                           results_by_threshold: Dict[float, Dict[str, Any]] = None) -> None:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        predictions: Dictionary of service -> method -> prediction data
        evaluation_results: Optional evaluation results
        results_by_threshold: Optional threshold analysis results
    """
    print("🎨 CREATING VISUALIZATION DASHBOARD")
    print("=" * 50)
    
    # 1. Confidence distribution
    if predictions:
        plot_confidence_distribution(predictions, "Prediction Confidence Distribution")
    
    # 2. Similarity vs Confidence scatter plot
    if predictions:
        plot_similarity_vs_confidence(predictions, "Similarity vs Confidence Analysis")
    
    # 3. Threshold analysis
    if results_by_threshold:
        plot_threshold_analysis(results_by_threshold, "Threshold Impact Analysis")
    
    # 4. Label distribution
    if predictions:
        label_counts = {}
        for service, methods in predictions.items():
            service_counts = defaultdict(int)
            for method, data in methods.items():
                if isinstance(data, dict):
                    service_counts[data['label']] += 1
            label_counts[service] = dict(service_counts)
        
        if label_counts:
            plot_label_distribution(label_counts, "Label Distribution by Service")
    
    print("✅ Dashboard visualization complete!")
