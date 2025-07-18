"""Evaluation and analysis utilities for label propagation."""

import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import config
from visualization import (
    plot_confusion_matrix,
    print_service_predictions_summary,
)


class Evaluator:
    def __init__(self, data_manager):
        """
        Initialize the evaluator.
        
        Args:
            data_manager: DataManager instance with loaded embeddings and labels
        """
        self.data_manager = data_manager
    
    def evaluate_propagation(self, test_size: float = 0.3, k: int = 5) -> Dict[str, float]:
        """
        Evaluate label propagation using train/test split of manually labeled data.
        Uses sklearn for evaluation to maintain compatibility with existing metrics.
        """
        if len(self.data_manager.method_labels) < 10:
            print("⚠️ Not enough labeled data for evaluation")
            return {}
        
        # Prepare data
        methods = list(self.data_manager.method_labels.keys())
        X = np.array([self.data_manager.method_embeddings[method] for method in methods])
        y = [self.data_manager.method_labels[method] for method in methods]
        
        # Split data
        X_train, X_test, y_train, y_test, methods_train, methods_test = train_test_split(
            X, y, methods, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train, y_train)
        
        # Predict
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("🎯 Evaluation Results:")
        print(classification_report(y_test, y_pred))
        
        # Use visualization utility for confusion matrix
        plot_confusion_matrix(y_test, y_pred, 'Label Propagation - Confusion Matrix')
        
        return report
    
    def find_optimal_k(self, k_values: List[int] = None) -> int:
        """Find optimal k value using cross-validation."""
        if k_values is None:
            k_values = config.K_VALUES_TO_TEST
            
        if len(self.data_manager.method_labels) < 10:
            print("⚠️ Not enough labeled data for k optimization")
            return 5
        
        methods = list(self.data_manager.method_labels.keys())
        X = np.array([self.data_manager.method_embeddings[method] for method in methods])
        y = [self.data_manager.method_labels[method] for method in methods]
        
        best_k = 5
        best_score = 0
        
        print("🔍 Finding optimal k value:")
        for k in k_values:
            if k < len(set(y)):  # Ensure k is less than number of classes
                knn = KNeighborsClassifier(n_neighbors=min(k, len(X)-1), metric='cosine')
                scores = cross_val_score(knn, X, y, cv=min(5, len(X)//2), scoring='f1_macro')
                avg_score = scores.mean()
                print(f"  k={k}: F1-score = {avg_score:.3f} ± {scores.std():.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k
        
        print(f"✅ Best k value: {best_k} (F1-score: {best_score:.3f})")
        return best_k
    
    def analyze_label_distribution(self) -> None:
        """Analyze the distribution of labels across services."""
        service_label_counts = defaultdict(lambda: defaultdict(int))
        
        for (service, method), label in self.data_manager.method_labels.items():
            service_label_counts[service][label] += 1
        
        print("📊 Label Distribution by Service:")
        print("-" * 50)
        
        for service, label_counts in service_label_counts.items():
            total = sum(label_counts.values())
            print(f"{service.upper()}:")
            for label, count in label_counts.items():
                percentage = (count / total) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
            print()
    
    def analyze_method_coverage(self) -> None:
        """Analyze method coverage across services."""
        print("📈 Method Coverage Analysis:")
        print("-" * 50)
        
        for service, methods in self.data_manager.service_methods.items():
            labeled_count = len([m for m in methods if (service, m) in self.data_manager.method_labels])
            total_count = len(methods)
            unlabeled_count = total_count - labeled_count
            coverage = (labeled_count / total_count) * 100 if total_count > 0 else 0
            
            print(f"{service.upper()}:")
            print(f"📊 {labeled_count}/{total_count} labeled ({coverage:.1f}% coverage)")
            print(f"🔍 {unlabeled_count} methods need propagation")
        print()
    
    def generate_summary(self, final_predictions: Dict, cross_service_predictions: Dict,
                        evaluation_results: Dict, best_k: int) -> Dict:
        """Generate comprehensive experiment summary."""
        summary = {
            'experiment_metadata': {
                'optimal_k': best_k,
                'within_service_threshold': config.DEFAULT_WITHIN_SERVICE_THRESHOLD,
                'cross_service_threshold': config.DEFAULT_CROSS_SERVICE_THRESHOLD,
                'labeled_services': config.LABELED_SERVICES,
                'total_labeled_methods': len(self.data_manager.method_labels),
                'evaluation_results': evaluation_results
            },
            'within_service_results': {
                'total_predictions': sum(len(pred) for pred in final_predictions.values()),
                'predictions_by_service': {}
            },
            'cross_service_results': {
                'total_predictions': sum(len(pred) for pred in cross_service_predictions.values()),
                'successful_transfers': list(cross_service_predictions.keys())
            },
            'recommendations': {
                'high_confidence_services': [],
                'manual_review_needed': [],
                'next_similar_services': {}
            }
        }
        
        # Analyze within-service results
        for service, predictions in final_predictions.items():
            service_summary = self._analyze_service_predictions(predictions)
            summary['within_service_results']['predictions_by_service'][service] = service_summary
            
            # Add to recommendations
            if service_summary['avg_confidence'] > 0.7 and len(predictions) >= 5:
                summary['recommendations']['high_confidence_services'].append(service)
            
            low_conf_count = len(predictions) - service_summary['high_confidence_count']
            if low_conf_count > len(predictions) * 0.3:  # More than 30% low confidence
                summary['recommendations']['manual_review_needed'].append(service)
        
        # Add similar services recommendations
        for service in config.LABELED_SERVICES:
            if service.lower() in config.SIMILAR_SERVICES:
                summary['recommendations']['next_similar_services'][service] = \
                    config.SIMILAR_SERVICES[service.lower()]
        
        return summary
    
    def _analyze_service_predictions(self, predictions: Dict) -> Dict:
        """Analyze predictions for a single service."""
        service_summary = {
            'total_predictions': len(predictions),
            'label_distribution': {},
            'avg_confidence': 0,
            'high_confidence_count': 0
        }
        
        total_conf = 0
        for method, pred_data in predictions.items():
            label = pred_data['label']
            confidence = pred_data['confidence']
            
            service_summary['label_distribution'][label] = \
                service_summary['label_distribution'].get(label, 0) + 1
            total_conf += confidence
            
            if confidence > 0.8:
                service_summary['high_confidence_count'] += 1
        
        if len(predictions) > 0:
            service_summary['avg_confidence'] = total_conf / len(predictions)
        
        return service_summary
    
    def print_final_summary(self, summary: Dict, final_predictions: Dict, 
                           cross_service_predictions: Dict) -> None:
        """Print final results and recommendations."""
        print("\n" + "="*50)
        print("FINAL RESULTS AND RECOMMENDATIONS")
        print("="*50)
        
        print(f"🎉 Label propagation complete!")
        print(f"📊 Within-service predictions: {summary['within_service_results']['total_predictions']}")
        if cross_service_predictions:
            print(f"🔀 Cross-service predictions: {summary['cross_service_results']['total_predictions']}")
        
        print("🎯 Next Steps Recommendations:")
        
        print("1. ✅ High-confidence services (ready for use):")
        for service in summary['recommendations']['high_confidence_services']:
            pred_count = len(final_predictions.get(service, {}))
            avg_conf = summary['within_service_results']['predictions_by_service'][service]['avg_confidence']
            print(f"   • {service}: {pred_count} predictions, {avg_conf:.3f} avg confidence")
        
        print("2. ⚠️  Services needing manual review:")
        for service in summary['recommendations']['manual_review_needed']:
            pred_count = len(final_predictions.get(service, {}))
            print(f"   • {service}: {pred_count} predictions (review low-confidence ones)")
        
        print("3. 🔀 Recommended similar services for cross-service propagation:")
        for source_service, similar_list in summary['recommendations']['next_similar_services'].items():
            if similar_list:
                print(f"   • {source_service} → {', '.join(similar_list)}")
