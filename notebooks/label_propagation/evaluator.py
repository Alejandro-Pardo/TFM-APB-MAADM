"""Evaluation and analysis utilities for label propagation."""

import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import config
from visualization import plot_confusion_matrix


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
        
        # DEBUG: Print conflicting methods (can be deleted later)
        print("🔍 DEBUG - Conflicting Methods:")
        print("-" * 50)
        conflicts = []
        for i, (actual, predicted, method) in enumerate(zip(y_test, y_pred, methods_test)):
            if actual != predicted:
                conflicts.append({
                    'method': method,
                    'actual': actual,
                    'predicted': predicted
                })
        
        if conflicts:
            print(f"Found {len(conflicts)} conflicting predictions:")
            for conflict in conflicts:
                method_key = conflict['method']
                service = method_key[0] if isinstance(method_key, tuple) else str(method_key).split('.')[0]
                method_name = method_key[1] if isinstance(method_key, tuple) else str(method_key)
                print(f"  • {service}.{method_name}")
                print(f"    Actual: {conflict['actual']}")
                print(f"    Predicted: {conflict['predicted']}")
        else:
            print("No conflicts found - all predictions match actual labels!")
        print("-" * 50)
        # END DEBUG
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("🎯 Evaluation Results:")
        print(classification_report(y_test, y_pred))
        
        # Use visualization utility for confusion matrix
        plot_confusion_matrix(y_test, y_pred, 'Label Propagation - Confusion Matrix')
        
        return report
    
    def find_optimal_k(self, k_values: List[int] = None) -> int:
        """Find optimal k value using cross-validation - FINAL FIXED VERSION."""
        if k_values is None:
            k_values = config.K_VALUES_TO_TEST
            
        if len(self.data_manager.method_labels) < 10:
            print("⚠️ Not enough labeled data for k optimization")
            return 5
        
        methods = list(self.data_manager.method_labels.keys())
        X = np.array([self.data_manager.method_embeddings[method] for method in methods])
        y = [self.data_manager.method_labels[method] for method in methods]
        
        # Debug information
        print(f"📊 Data summary:")
        print(f"   • Total samples: {len(X)}")
        print(f"   • Embedding dimension: {X.shape[1] if len(X) > 0 else 'N/A'}")
        print(f"   • Unique labels: {set(y)}")
        print(f"   • Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Check for potential issues
        if len(set(y)) <= 1:
            print("⚠️ Only one class found - cannot perform classification")
            return 5
        
        if len(X) < 6:  # Need at least 6 samples for cross-validation
            print("⚠️ Too few samples for reliable cross-validation")
            return 5
        
        best_k = 5
        best_score = 0
        valid_scores_found = False
        
        print("🔍 Finding optimal k value:")
        
        # Calculate minimum samples per class for stratified CV
        min_class_size = min([list(y).count(label) for label in set(y)])
        max_cv_folds = min(5, min_class_size)  # Can't have more folds than smallest class
        
        print(f"   • Min class size: {min_class_size}")
        print(f"   • Max CV folds: {max_cv_folds}")
        
        if max_cv_folds < 2:
            print("⚠️ Some classes have too few samples for cross-validation")
            return 5
        
        for k in k_values:
            # The only constraint should be that k < total samples
            if k >= len(X):
                print(f"   k={k}: Skipped (k >= total samples)")
                continue
            
            if k < 1:
                print(f"   k={k}: Skipped (k < 1)")
                continue
            
            try:
                knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
                
                # Use stratified CV with appropriate number of folds
                cv = StratifiedKFold(n_splits=max_cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(knn, X, y, cv=cv, scoring='f1_macro')
                
                avg_score = scores.mean()
                std_score = scores.std()
                
                print(f"   k={k}: F1-score = {avg_score:.3f} ± {std_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k
                    valid_scores_found = True
                    
            except Exception as e:
                print(f"   k={k}: Error during evaluation - {str(e)}")
                continue
        
        if not valid_scores_found:
            print("⚠️ No valid k values found, using default k=3")
            best_k = 3
            best_score = 0.0
        
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
                'service_similarity_threshold': config.DEFAULT_SERVICE_THRESHOLD,
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
        

    def compare_cross_service_predictions(self, group_predictions: Dict, all_to_all_predictions: Dict) -> Dict:
        """
        Compare group-based cross-service predictions with all-to-all predictions.
        
        Args:
            group_predictions: Results from group-based cross-service propagation
            all_to_all_predictions: Results from all-to-all cross-service propagation
            
        Returns:
            Dictionary with comparison statistics
        """
        
        comparison_stats = {
            'agreement_stats': {},
            'disagreement_details': {},
            'coverage_comparison': {},
            'confidence_comparison': {},
            'summary': {}
        }
        
        # Flatten group predictions for easier comparison
        group_flat = {}
        for group_name, group_data in group_predictions.items():
            for service, service_predictions in group_data.items():
                if service not in group_flat:
                    group_flat[service] = {}
                group_flat[service].update(service_predictions)
        
        # Compare each service
        for service in set(list(group_flat.keys()) + list(all_to_all_predictions.keys())):
            group_preds = group_flat.get(service, {})
            all_preds = all_to_all_predictions.get(service, {})
            
            # Find common methods
            common_methods = set(group_preds.keys()) & set(all_preds.keys())
            
            if common_methods:
                agreements = 0
                disagreements = []
                group_confidences = []
                all_confidences = []
                
                for method in common_methods:
                    group_pred = group_preds[method]
                    all_pred = all_preds[method]
                    
                    group_label = group_pred['label'] if isinstance(group_pred, dict) else group_pred
                    all_label = all_pred['label'] if isinstance(all_pred, dict) else all_pred
                    
                    if group_label == all_label:
                        agreements += 1
                    else:
                        disagreements.append({
                            'method': method,
                            'group_prediction': group_label,
                            'all_to_all_prediction': all_label,
                            'group_confidence': group_pred.get('confidence', 0) if isinstance(group_pred, dict) else 0,
                            'all_confidence': all_pred.get('confidence', 0) if isinstance(all_pred, dict) else 0
                        })
                    
                    if isinstance(group_pred, dict) and 'confidence' in group_pred:
                        group_confidences.append(group_pred['confidence'])
                    if isinstance(all_pred, dict) and 'confidence' in all_pred:
                        all_confidences.append(all_pred['confidence'])
                
                agreement_rate = agreements / len(common_methods) if common_methods else 0
                
                comparison_stats['agreement_stats'][service] = {
                    'total_common_methods': len(common_methods),
                    'agreements': agreements,
                    'disagreements': len(disagreements),
                    'agreement_rate': agreement_rate
                }
                
                comparison_stats['disagreement_details'][service] = disagreements
                
                comparison_stats['confidence_comparison'][service] = {
                    'group_avg_confidence': np.mean(group_confidences) if group_confidences else 0,
                    'all_to_all_avg_confidence': np.mean(all_confidences) if all_confidences else 0
                }
            
            # Coverage comparison
            comparison_stats['coverage_comparison'][service] = {
                'group_predictions': len(group_preds),
                'all_to_all_predictions': len(all_preds),
                'group_only': len(set(group_preds.keys()) - set(all_preds.keys())),
                'all_to_all_only': len(set(all_preds.keys()) - set(group_preds.keys())),
                'common_methods': len(set(group_preds.keys()) & set(all_preds.keys()))
            }
        
        # Overall summary
        total_agreements = sum(stats['agreements'] for stats in comparison_stats['agreement_stats'].values())
        total_common = sum(stats['total_common_methods'] for stats in comparison_stats['agreement_stats'].values())
        overall_agreement = total_agreements / total_common if total_common > 0 else 0
        
        total_group_predictions = sum(len(preds) for preds in group_flat.values())
        total_all_predictions = sum(len(preds) for preds in all_to_all_predictions.values())
        
        comparison_stats['summary'] = {
            'overall_agreement_rate': overall_agreement,
            'total_common_predictions': total_common,
            'total_agreements': total_agreements,
            'total_disagreements': total_common - total_agreements,
            'group_total_predictions': total_group_predictions,
            'all_to_all_total_predictions': total_all_predictions
        }
        
        # Print summary
        print(f"📊 Overall Agreement Rate: {overall_agreement:.3f}")
        print(f"🤝 Total Agreements: {total_agreements} / {total_common}")
        print(f"📈 Group-based predictions: {total_group_predictions}")
        print(f"🌍 All-to-all predictions: {total_all_predictions}")
        
        print("\n📋 Service-level Agreement Rates:")
        for service, stats in comparison_stats['agreement_stats'].items():
            print(f"   • {service}: {stats['agreement_rate']:.3f} "
                  f"({stats['agreements']}/{stats['total_common_methods']})")
        
        if total_common - total_agreements > 0:
            print(f"\n⚠️  Found {total_common - total_agreements} disagreements across all services")
            print("   Review disagreement_details in the returned comparison stats for specific conflicts")
        
        return comparison_stats
