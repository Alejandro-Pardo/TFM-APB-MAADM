"""Core label propagation algorithms for AWS API security classification."""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from annoy import AnnoyIndex
import config


class LabelPropagator:
    def __init__(self, data_manager):
        """
        Initialize the label propagator.
        
        Args:
            data_manager: DataManager instance with loaded embeddings and labels
        """
        self.data_manager = data_manager
    
    def get_neighbors_with_similarities(self, service: str, query_embedding: np.ndarray, 
                                      k: int) -> List[Tuple[Tuple[str, str], float]]:
        """Get k nearest neighbors with cosine similarities."""
        self.data_manager.ensure_service_index(service)
        
        index = self.data_manager.annoy_indexes[service]
        method_lookup = self.data_manager.method_lookups[service]
        
        # Get more neighbors than needed since we'll filter
        search_k = min(k * 3, len(method_lookup))
        neighbor_indices = index.get_nns_by_vector(query_embedding, search_k)
        
        # Calculate actual cosine similarities
        neighbors_with_sims = []
        for idx in neighbor_indices:
            if idx in method_lookup:
                neighbor_key = method_lookup[idx]
                neighbor_embedding = self.data_manager.method_embeddings[neighbor_key]
                
                # Calculate cosine similarity
                similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
                neighbors_with_sims.append((neighbor_key, similarity))
        
        # Sort by similarity and return top k
        neighbors_with_sims.sort(key=lambda x: x[1], reverse=True)
        return neighbors_with_sims[:k]
    
    def propagate_within_service(self, service: str, k: int = 5, threshold: float = 0.7, 
                            max_iterations: int = 10, min_confidence: float = 0.5, 
                            min_threshold: float = 0.1, save_history: bool = False,
                            verbose: bool = True, embedding_format: str = 'with_params') -> Dict[str, str]:
        """
        Iteratively propagate labels within a single service.
        
        Args:
            service: Service name to propagate within
            k: Number of neighbors for k-NN
            threshold: Minimum similarity threshold for high-confidence propagation
            max_iterations: Maximum number of iterations to perform
            min_confidence: Minimum confidence for accepting predictions
            min_threshold: Minimum threshold to lower to
            save_history: Whether to save iteration history to history.json
            verbose: Whether to print progress information
            embedding_format: Format of embeddings to use ('with_params', 'method_only', etc.)
            
        Returns:
            Dictionary of method -> predicted label
        """
        service = service.lower()
        predictions = {}
        
        # Load embeddings with specified format if different from current
        if embedding_format != 'with_params':
            temp_embeddings = self.data_manager.get_service_embeddings(service, embedding_format)
            # Temporarily replace embeddings
            original_embeddings = {}
            for key in temp_embeddings:
                if key in self.data_manager.method_embeddings:
                    original_embeddings[key] = self.data_manager.method_embeddings[key]
                self.data_manager.method_embeddings[key] = temp_embeddings[key]
        
        # Get labeled and unlabeled methods for this service
        labeled_methods = [(svc, method) for (svc, method) in self.data_manager.method_labels.keys() 
                        if svc == service]
        
        # Find the original case service name for service_methods lookup
        service_original = None
        for service_key in self.data_manager.service_methods.keys():
            if service_key.lower() == service:
                service_original = service_key
                break
        
        if not service_original:
            if verbose:
                print(f"⚠️ Service not found: {service}")
            return predictions
            
        all_methods = [(service, method) for method in self.data_manager.service_methods[service_original]]
        unlabeled_methods = [m for m in all_methods if m not in self.data_manager.method_labels]
        
        if len(labeled_methods) == 0:
            if verbose:
                print(f"⚠️ No labeled methods found for service: {service}")
            return predictions
        
        if verbose:
            print(f"\n🔄 Propagating in {service}: {len(labeled_methods)} labeled → {len(unlabeled_methods)} unlabeled")
        
        # Initialize history tracking (only if save_history is True)
        history = None
        if save_history:
            history = {
                'service': service,
                'iterations': []
            }
            
            # Add iteration 0 (initial labeled methods)
            initial_labeled = {}
            for method_key in labeled_methods:
                method_name = method_key[1]  # Extract just the method name
                label = self.data_manager.method_labels[method_key]
                initial_labeled[method_name] = {
                    'label': label,
                    'iteration': 0,
                    'initial': True
                }
            
            history['iterations'].append({
                'iteration': 0,
                'threshold': threshold,
                'newly_labeled': initial_labeled,
                'total_labeled_count': len(initial_labeled)
            })
        
        # Create a copy of method_labels to track temporary labels during iteration
        temp_method_labels = self.data_manager.method_labels.copy()
        remaining_unlabeled = set(unlabeled_methods)
        
        # Initialize threshold
        current_threshold = threshold
        
        for iteration in range(max_iterations):
            if not remaining_unlabeled:
                break
                
            iteration_predictions = {}
            
            if verbose:
                print(f"  Iteration {iteration + 1}: {len(remaining_unlabeled)} remaining (threshold: {current_threshold:.2f})")
            
            # Predict for remaining unlabeled methods
            for method_key in list(remaining_unlabeled):
                if method_key in self.data_manager.method_embeddings:
                    embedding = self.data_manager.method_embeddings[method_key]
                    
                    # Get neighbors with similarities
                    neighbors_with_sims = self.get_neighbors_with_similarities(service, embedding, k)
                    
                    # Filter by labeled neighbors (including temporary labels)
                    labeled_neighbors = []
                    for neighbor_key, similarity in neighbors_with_sims:
                        if neighbor_key in temp_method_labels:
                            labeled_neighbors.append((neighbor_key, similarity))
                    
                    if labeled_neighbors:
                        # Filter by threshold
                        valid_neighbors = [(nk, sim) for nk, sim in labeled_neighbors 
                                        if sim >= current_threshold]
                        
                        if valid_neighbors or iteration == max_iterations - 1:
                            # Use valid neighbors or all neighbors on final iteration
                            neighbors_to_use = valid_neighbors if valid_neighbors else labeled_neighbors
                            
                            label_weights = defaultdict(float)
                            label_counts = defaultdict(int)
                            for neighbor_key, similarity in neighbors_to_use:
                                label = temp_method_labels[neighbor_key]
                                label_weights[label] += similarity
                                label_counts[label] += 1
                            
                            predicted_label = max(label_weights, key=label_weights.get)
                            total_neighbors = len(neighbors_to_use)
                            confidence = label_counts[predicted_label] / total_neighbors
                            max_similarity = max(sim for _, sim in neighbors_to_use)
                            
                            # Accept if confidence meets minimum or it's the final iteration
                            if confidence >= min_confidence or iteration == max_iterations - 1:
                                iteration_predictions[method_key[1]] = {
                                    'label': predicted_label,
                                    'confidence': confidence,
                                    'similarity': max_similarity,
                                    'neighbors': [neighbor_key for neighbor_key, _ in neighbors_to_use],
                                    'iteration': iteration + 1
                                }
            
            # Add iteration to history (only if save_history is True)
            if save_history and history:
                newly_labeled_for_history = {}
                if iteration_predictions:
                    for method_name, pred_data in iteration_predictions.items():
                        newly_labeled_for_history[method_name] = {
                            'label': pred_data['label'],
                            'iteration': iteration + 1,
                            'initial': False
                        }
                
                # Calculate total labeled so far
                total_labeled = len(initial_labeled) + len([p for p in predictions.values()])
                if iteration_predictions:
                    total_labeled += len(iteration_predictions)
                
                history['iterations'].append({
                    'iteration': iteration + 1,
                    'threshold': current_threshold,
                    'newly_labeled': newly_labeled_for_history,
                    'total_labeled_count': total_labeled
                })
            
            # Add new predictions
            if iteration_predictions:
                predictions.update(iteration_predictions)
                if verbose:
                    print(f"    ✅ Added {len(iteration_predictions)} predictions")
                
                # Update temporary labels for next iteration
                for method_name, pred_data in iteration_predictions.items():
                    method_key = (service, method_name)
                    temp_method_labels[method_key] = pred_data['label']
                    remaining_unlabeled.discard(method_key)
            else:
                # Lower threshold for next iteration
                if iteration < max_iterations - 1:
                    current_threshold = max(min_threshold, current_threshold - 0.1)
                    if current_threshold == min_threshold and verbose:
                        print(f"    📉 Minimum threshold {min_threshold:.2f} reached")
                        break
        
        # Save history to JSON file (only if save_history is True)
        if save_history and history:
            self._save_history_to_file(history)
        
        if verbose:
            if remaining_unlabeled:
                print(f"  ⚠️ {len(remaining_unlabeled)} methods remain unlabeled")
            else:
                print(f"  ✅ All methods labeled!")
        
        # Restore original embeddings if changed
        if embedding_format != 'with_params':
            for key in original_embeddings:
                self.data_manager.method_embeddings[key] = original_embeddings[key]
        
        return predictions

    def _save_history_to_file(self, history):
        """Save iteration history to JSON file."""
        try:
            history_file = config.HISTORY_FILE
            if history_file.exists():
                with open(history_file, 'r') as f:
                    all_history = json.load(f)
            else:
                all_history = {}
            
            all_history[history['service']] = history
            
            with open(history_file, 'w') as f:
                json.dump(all_history, f, indent=2, default=str)
            
        except Exception as e:
            print(f"⚠️ Failed to save history: {e}")

    def propagate_all_services(self, k: int = 5, threshold: float = 0.7, 
                            max_iterations: int = 5, min_confidence: float = 0.5, 
                            min_threshold: float = 0.1, save_history: bool = False,
                            verbose: bool = True, embedding_format: str = 'with_params') -> Dict[str, Dict[str, str]]:
        """Propagate labels for all loaded services."""
        all_predictions = {}
        
        if verbose:
            print(f"\n📊 Starting within-service propagation for {len(self.data_manager.service_methods)} services")
            print(f"   Parameters: k={k}, threshold={threshold:.2f}→{min_threshold:.2f}, format={embedding_format}")
        
        for service in self.data_manager.service_methods.keys():
            predictions = self.propagate_within_service(
                service, k, threshold, max_iterations, min_confidence, 
                min_threshold, save_history, verbose, embedding_format
            )
            if predictions:
                all_predictions[service] = predictions
        
        if verbose:
            total = sum(len(pred) for pred in all_predictions.values())
            print(f"\n✅ Within-service complete: {total} total predictions across {len(all_predictions)} services")
        
        return all_predictions

    def propagate_group_cross_service(self, group_name: str, group_config: Dict, 
                                    k: int = 5, threshold: float = 0.6,
                                    min_threshold: float = 0.3, min_confidence: float = 0.5,
                                    within_service_predictions: Dict = None,
                                    verbose: bool = True, embedding_format: str = 'method_only') -> Dict[str, Dict[str, str]]:
        """
        Propagate labels from core services to target services within a group with adaptive thresholding.
        
        Args:
            group_name: Name of the service group
            group_config: Configuration dict with 'core_services' and 'target_services'
            k: Number of neighbors for k-NN
            threshold: Initial similarity threshold for propagation
            min_threshold: Minimum threshold to lower to
            min_confidence: Minimum confidence for accepting predictions
            within_service_predictions: Optional dict of within-service predictions to include as labels
            verbose: Whether to print progress information
            embedding_format: Format of embeddings to use for cross-service comparison
            
        Returns:
            Dictionary of {target_service: {method: prediction_data}}
        """
        core_services = [s.lower() for s in group_config['core_services']]
        target_services = [s.lower() for s in group_config['target_services']]
        
        if verbose:
            print(f"\n🌐 Group: {group_name}")
            print(f"   Core: {core_services}, Target: {target_services}")
        
        group_predictions = {}
        
        # Collect all labeled methods from core services (manual + within-service predictions)
        core_labeled_methods = {}
        
        # Add manual labels from core services
        for method_key, label in self.data_manager.method_labels.items():
            service, method = method_key
            if service in core_services:
                core_labeled_methods[method_key] = label
        
        # Add within-service predictions if provided
        if within_service_predictions:
            for service, service_predictions in within_service_predictions.items():
                if service.lower() in core_services:
                    for method_name, pred_data in service_predictions.items():
                        method_key = (service.lower(), method_name)
                        if isinstance(pred_data, dict) and 'label' in pred_data:
                            core_labeled_methods[method_key] = pred_data['label']
                        else:
                            core_labeled_methods[method_key] = pred_data
        
        if not core_labeled_methods:
            if verbose:
                print(f"   ⚠️ No labeled methods found in core services")
            return group_predictions
        
        if verbose:
            print(f"   📊 {len(core_labeled_methods)} labeled methods in core")
        
        # Get embeddings with specified format
        available_core_services = []
        available_target_services = []
        
        for service_key in self.data_manager.service_methods.keys():
            if service_key.lower() in core_services:
                available_core_services.append(service_key)
            elif service_key.lower() in target_services:
                available_target_services.append(service_key)
        
        if not available_target_services:
            if verbose:
                print(f"   ⚠️ No target services available")
            return group_predictions
        
        all_services = available_core_services + available_target_services
        
        # Create combined embeddings for all services in the group
        combined_embeddings = {}
        for service in all_services:
            service_embeddings = self.data_manager.get_service_embeddings(service, embedding_format)
            combined_embeddings.update(service_embeddings)
        
        # Propagate to each target service with adaptive thresholding
        for target_service in available_target_services:
            target_methods = [(target_service.lower(), method) for method in self.data_manager.service_methods[target_service]]
            service_predictions = {}
            
            # Create a copy of core labeled methods that will be updated with new predictions
            temp_labeled_methods = core_labeled_methods.copy()
            
            # Build initial index with core labeled methods
            core_index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
            core_lookup = {}
            
            idx = 0
            for method_key in temp_labeled_methods.keys():
                if method_key in combined_embeddings:
                    core_index.add_item(idx, combined_embeddings[method_key])
                    core_lookup[idx] = method_key
                    idx += 1
            
            if idx == 0:
                if verbose:
                    print(f"   ⚠️ No embeddings found for {target_service}")
                continue
            
            core_index.build(config.ANNOY_N_TREES)
            
            # Adaptive threshold for this service
            current_threshold = threshold
            unlabeled_methods = [m for m in target_methods if m not in self.data_manager.method_labels]
            
            if verbose and unlabeled_methods:
                print(f"   🎯 {target_service}: {len(unlabeled_methods)} methods to predict")
            
            iterations = 0
            
            while unlabeled_methods:
                iteration_predictions = {}
                
                for method_key in unlabeled_methods:
                    if method_key in combined_embeddings:
                        embedding = combined_embeddings[method_key]
                        
                        # Get neighbors from current labeled set (includes newly labeled)
                        neighbor_indices = core_index.get_nns_by_vector(embedding, k)
                        
                        # Calculate similarities
                        all_neighbors = []
                        for idx in neighbor_indices:
                            if idx in core_lookup:
                                neighbor_key = core_lookup[idx]
                                neighbor_embedding = combined_embeddings[neighbor_key]
                                similarity = cosine_similarity([embedding], [neighbor_embedding])[0][0]
                                all_neighbors.append((neighbor_key, similarity))
                        
                        if all_neighbors:
                            # Filter by threshold
                            valid_neighbors = [(nk, sim) for nk, sim in all_neighbors if sim >= current_threshold]
                            
                            if valid_neighbors:
                                label_weights = defaultdict(float)
                                label_counts = defaultdict(int)
                                for neighbor_key, similarity in valid_neighbors:
                                    label = temp_labeled_methods[neighbor_key]
                                    label_weights[label] += similarity
                                    label_counts[label] += 1
                                
                                predicted_label = max(label_weights, key=label_weights.get)
                                total_neighbors = len(valid_neighbors)
                                confidence = label_counts[predicted_label] / total_neighbors
                                max_similarity = max(sim for _, sim in valid_neighbors)
                                
                                # Accept if confidence meets minimum
                                if confidence >= min_confidence:
                                    iteration_predictions[method_key] = {
                                        'label': predicted_label,
                                        'confidence': confidence,
                                        'similarity': max_similarity,
                                        'core_neighbors': [f"{nk[0]}.{nk[1]}" for nk, _ in valid_neighbors[:3]],
                                        'threshold_used': current_threshold,
                                        'group': group_name,
                                        'iteration': iterations + 1
                                    }
                
                # Update predictions and labeled methods
                if iteration_predictions:
                    # Add to service predictions (only store method name as key)
                    for method_key, pred_data in iteration_predictions.items():
                        service_predictions[method_key[1]] = pred_data
                        # Add to temporary labeled methods for next iteration
                        temp_labeled_methods[method_key] = pred_data['label']
                    
                    # Rebuild index with newly labeled methods
                    core_index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
                    core_lookup = {}
                    idx = 0
                    for method_key in temp_labeled_methods.keys():
                        if method_key in combined_embeddings:
                            core_index.add_item(idx, combined_embeddings[method_key])
                            core_lookup[idx] = method_key
                            idx += 1
                    core_index.build(config.ANNOY_N_TREES)
                    
                    # Update unlabeled list
                    unlabeled_methods = [m for m in unlabeled_methods 
                                        if m[1] not in service_predictions]
                    
                    if verbose:
                        print(f"      Iteration {iterations+1}: {len(iteration_predictions)} predictions (threshold: {current_threshold:.2f})")
                else:
                    # No predictions made at current threshold, try lowering threshold
                    if current_threshold > min_threshold + 0.01:  # Small epsilon to avoid float comparison issues
                        current_threshold = max(min_threshold, current_threshold - 0.1)
                        if verbose:
                            print(f"      Lowering threshold to {current_threshold:.2f}")
                    else:
                        if verbose:
                            print(f"      Minimum threshold reached ({min_threshold:.2f})")
                        break
                
                iterations += 1
            
            if service_predictions:
                group_predictions[target_service.lower()] = service_predictions
                if verbose:
                    remaining_count = len(unlabeled_methods)
                    print(f"      ✅ {len(service_predictions)} predictions")
                    if remaining_count > 0:
                        print(f"      ⚠️ {remaining_count} methods remain unlabeled")
            elif verbose and unlabeled_methods:
                print(f"      ❌ No predictions made ({len(unlabeled_methods)} methods remain unlabeled)")
        
        return group_predictions

    def propagate_all_groups_cross_service(self, k: int = 5, threshold: float = 0.5,
                                         min_threshold: float = 0.3, min_confidence: float = 0.5,
                                         within_service_predictions: Dict = None,
                                         verbose: bool = True, embedding_format: str = 'method_only') -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Propagate labels for all defined service groups with adaptive thresholding.
        """
        all_group_predictions = {}
        
        if verbose:
            print(f"\n📊 Starting group-based cross-service propagation")
            print(f"   Parameters: k={k}, threshold={threshold:.2f}→{min_threshold:.2f}, format={embedding_format}")
        
        for group_name, group_config in config.CROSS_SERVICE_GROUPS.items():
            group_predictions = self.propagate_group_cross_service(
                group_name, group_config, k, threshold, min_threshold, 
                min_confidence, within_service_predictions, verbose, embedding_format
            )
            
            if group_predictions:
                all_group_predictions[group_name] = group_predictions
        
        if verbose:
            total = sum(
                len(service_preds) 
                for group_data in all_group_predictions.values() 
                for service_preds in group_data.values()
            )
            print(f"\n✅ Group cross-service complete: {total} total predictions across {len(all_group_predictions)} groups")
        
        return all_group_predictions

    def propagate_all_to_all_cross_service(self, k: int = 5, threshold: float = 0.3,
                                         min_threshold: float = 0.2, min_confidence: float = 0.5,
                                         within_service_predictions: Dict = None,
                                         verbose: bool = True, embedding_format: str = 'method_only') -> Dict[str, Dict[str, str]]:
        """
        Propagate labels from ALL services with labels to ALL other services with adaptive thresholding.
        """
        if verbose:
            print(f"\n📊 Starting all-to-all cross-service propagation")
            print(f"   Parameters: k={k}, threshold={threshold:.2f}→{min_threshold:.2f}, format={embedding_format}")
        
        all_predictions = {}
        
        # Collect all labeled methods (manual + within-service predictions)
        all_labeled_methods = {}
        all_labeled_methods.update(self.data_manager.method_labels)
        
        if within_service_predictions:
            for service, service_predictions in within_service_predictions.items():
                for method_name, pred_data in service_predictions.items():
                    method_key = (service.lower(), method_name)
                    if isinstance(pred_data, dict) and 'label' in pred_data:
                        all_labeled_methods[method_key] = pred_data['label']
                    else:
                        all_labeled_methods[method_key] = pred_data
        
        if not all_labeled_methods:
            if verbose:
                print("   ⚠️ No labeled methods found")
            return all_predictions
        
        if verbose:
            print(f"   📊 Total labeled methods: {len(all_labeled_methods)}")
        
        # Get all available services
        all_services = list(self.data_manager.service_methods.keys())
        
        # Create combined embeddings for all services
        combined_embeddings = {}
        for service in all_services:
            service_embeddings = self.data_manager.get_service_embeddings(service, embedding_format)
            combined_embeddings.update(service_embeddings)
        
        # Create index for all labeled methods
        labeled_index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
        labeled_lookup = {}
        
        idx = 0
        for method_key in all_labeled_methods.keys():
            if method_key in combined_embeddings:
                labeled_index.add_item(idx, combined_embeddings[method_key])
                labeled_lookup[idx] = method_key
                idx += 1
        
        if idx == 0:
            if verbose:
                print("   ⚠️ No embeddings found for labeled methods")
            return all_predictions
        
        labeled_index.build(config.ANNOY_N_TREES)
        
        # Propagate to all services with adaptive thresholding
        for target_service in all_services:
            target_methods = [(target_service.lower(), method) for method in self.data_manager.service_methods[target_service]]
            service_predictions = {}
            
            # Filter unlabeled methods
            unlabeled_methods = []
            for method_key in target_methods:
                if (method_key not in self.data_manager.method_labels and 
                    not (within_service_predictions and target_service.lower() in within_service_predictions and 
                         method_key[1] in within_service_predictions[target_service.lower()])):
                    unlabeled_methods.append(method_key)
            
            if not unlabeled_methods:
                continue
            
            if verbose:
                print(f"   🎯 {target_service}: {len(unlabeled_methods)} methods to predict")
            
            current_threshold = threshold
            iterations = 0
            
            while unlabeled_methods:
                iteration_predictions = {}
                
                for method_key in unlabeled_methods:
                    if method_key in combined_embeddings:
                        embedding = combined_embeddings[method_key]
                        
                        # Get neighbors from all labeled methods
                        neighbor_indices = labeled_index.get_nns_by_vector(embedding, k * 2)  # Get more candidates
                        
                        # Calculate similarities (excluding same service)
                        all_neighbors = []
                        for idx in neighbor_indices:
                            if idx in labeled_lookup:
                                neighbor_key = labeled_lookup[idx]
                                if neighbor_key[0] != target_service.lower():  # Skip same service
                                    neighbor_embedding = combined_embeddings[neighbor_key]
                                    similarity = cosine_similarity([embedding], [neighbor_embedding])[0][0]
                                    all_neighbors.append((neighbor_key, similarity))
                        
                        if all_neighbors:
                            # Filter by threshold and take top k
                            valid_neighbors = [(nk, sim) for nk, sim in all_neighbors if sim >= current_threshold][:k]
                            
                            if valid_neighbors:
                                label_weights = defaultdict(float)
                                label_counts = defaultdict(int)
                                for neighbor_key, similarity in valid_neighbors:
                                    label = all_labeled_methods[neighbor_key]
                                    label_weights[label] += similarity
                                    label_counts[label] += 1
                                
                                predicted_label = max(label_weights, key=label_weights.get)
                                total_neighbors = len(valid_neighbors)
                                confidence = label_counts[predicted_label] / total_neighbors
                                max_similarity = max(sim for _, sim in valid_neighbors)
                                
                                if confidence >= min_confidence:
                                    iteration_predictions[method_key[1]] = {
                                        'label': predicted_label,
                                        'confidence': confidence,
                                        'similarity': max_similarity,
                                        'source_neighbors': [f"{nk[0]}.{nk[1]}" for nk, _ in valid_neighbors[:3]],
                                        'threshold_used': current_threshold,
                                        'method': 'all_to_all'
                                    }
                
                # Update predictions
                if iteration_predictions:
                    service_predictions.update(iteration_predictions)
                    unlabeled_methods = [m for m in unlabeled_methods 
                                        if m[1] not in service_predictions]
                    if verbose:
                        print(f"      Iteration {iterations+1}: {len(iteration_predictions)} predictions (threshold: {current_threshold:.2f})")
                
                # Decide whether to continue or lower threshold
                if not unlabeled_methods:
                    break  # All methods labeled
                elif not iteration_predictions:
                    # No predictions made, try lowering threshold
                    if current_threshold > min_threshold:
                        current_threshold = max(min_threshold, current_threshold - 0.1)
                        if verbose:
                            print(f"      Lowering threshold to {current_threshold:.2f}")
                    else:
                        break  # Minimum threshold reached
                # Otherwise continue with same threshold to label remaining methods
                
                iterations += 1
            
            if service_predictions:
                all_predictions[target_service.lower()] = service_predictions
                if verbose:
                    remaining_count = len(unlabeled_methods)
                    print(f"      ✅ {len(service_predictions)} predictions")
                    if remaining_count > 0:
                        print(f"      ⚠️ {remaining_count} methods remain unlabeled")
            elif verbose and len(unlabeled_methods) > 0:
                print(f"      ❌ No predictions made ({len(unlabeled_methods)} methods remain unlabeled)")
        
        if verbose:
            total = sum(len(pred) for pred in all_predictions.values())
            print(f"\n✅ All-to-all complete: {total} total predictions")
        
        return all_predictions