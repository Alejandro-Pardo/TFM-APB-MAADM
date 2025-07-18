"""Core label propagation algorithms for AWS API security classification."""

import numpy as np
from typing import Dict, List, Tuple
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
                               min_threshold: float = 0.1) -> Dict[str, str]:
        """
        Iteratively propagate labels within a single service.
        Uses 'with_service_params' embeddings.
        
        Args:
            service: Service name to propagate within
            k: Number of neighbors for k-NN
            threshold: Minimum similarity threshold for high-confidence propagation
            max_iterations: Maximum number of iterations to perform
            min_confidence: Minimum confidence for accepting predictions
            min_threshold: Minimum threshold to lower to
            
        Returns:
            Dictionary of method -> predicted label
        """
        service = service.lower()
        predictions = {}
        
        # Get labeled and unlabeled methods for this service
        labeled_methods = [(svc, method) for (svc, method) in self.data_manager.method_labels.keys() 
                          if svc == service]
        all_methods = [(service, method) for method in self.data_manager.service_methods[service]]
        unlabeled_methods = [m for m in all_methods if m not in self.data_manager.method_labels]
        
        if len(labeled_methods) == 0:
            print(f"⚠️ No labeled methods found for service: {service}")
            return predictions
        
        print(f"🔄 Propagating in {service}: {len(labeled_methods)} labeled → {len(unlabeled_methods)} unlabeled")
        
        # Create a copy of method_labels to track temporary labels during iteration
        temp_method_labels = self.data_manager.method_labels.copy()
        remaining_unlabeled = set(unlabeled_methods)
        
        # Initialize threshold
        current_threshold = threshold
        
        for iteration in range(max_iterations):
            if not remaining_unlabeled:
                break
                
            iteration_predictions = {}
            
            print(f"🔄 Iteration {iteration + 1}: {len(remaining_unlabeled)} remaining unlabeled (threshold: {current_threshold:.1f})")
            
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
            
            # Add new predictions
            if iteration_predictions:
                predictions.update(iteration_predictions)
                print(f"✅ Added {len(iteration_predictions)} new predictions")
                
                # Update temporary labels for next iteration
                for method_name, pred_data in iteration_predictions.items():
                    method_key = (service, method_name)
                    temp_method_labels[method_key] = pred_data['label']
                    remaining_unlabeled.discard(method_key)
            else:
                print(f"⚠️ No new predictions in iteration {iteration + 1}")
                # Lower threshold for next iteration
                if iteration < max_iterations - 1:
                    current_threshold = max(min_threshold, current_threshold - 0.1)
                    if current_threshold == min_threshold:
                        print("⚠️ Minimum threshold reached, stopping further iterations")
                        break
                    print(f"📉 Lowering threshold to {current_threshold:.1f}")
        
        if remaining_unlabeled:
            print(f"⚠️ {len(remaining_unlabeled)} methods remain unlabeled after {iteration+1} iterations")
        else:
            print(f"🎉 All methods labeled after {iteration} iterations!")
        
        return predictions
    
    def propagate_all_services(self, k: int = 5, threshold: float = 0.7, 
                             max_iterations: int = 5, min_confidence: float = 0.5, 
                             min_threshold: float = 0.1) -> Dict[str, Dict[str, str]]:
        """Propagate labels for all loaded services."""
        all_predictions = {}
        
        for service in self.data_manager.service_methods.keys():
            predictions = self.propagate_within_service(
                service, k, threshold, max_iterations, min_confidence, min_threshold
            )
            if predictions:
                all_predictions[service] = predictions
                print(f"✅ {service}: {len(predictions)} predictions")
        
        return all_predictions
    
    def propagate_cross_service(self, source_service: str, target_service: str, 
                               k: int = 5, threshold: float = 0.6) -> Dict[str, str]:
        """
        Propagate labels from one service to another.
        Uses 'method_only' embeddings for cross-service comparison.
        
        Args:
            source_service: Service with labeled methods to use as training
            target_service: Service to predict labels for
            k: Number of neighbors for k-NN
            threshold: Minimum similarity threshold for propagation
            
        Returns:
            Dictionary of method -> predicted label for target service
        """
        source_service = source_service.lower()
        target_service = target_service.lower()
        predictions = {}
        
        # Get labeled methods from source service
        source_labeled = [(svc, method) for (svc, method) in self.data_manager.method_labels.keys() 
                         if svc == source_service]
        
        # Get all methods from target service
        target_methods = [(target_service, method) for method in self.data_manager.service_methods[target_service]]
        
        if len(source_labeled) == 0:
            print(f"⚠️  No labeled methods found for source service: {source_service}")
            return predictions
        
        print(f"🔀 Cross-service propagation: {source_service} ({len(source_labeled)} labeled) → "
              f"{target_service} ({len(target_methods)} methods)")
        
        # Get embeddings with 'method_only' format for cross-service
        source_embeddings, target_embeddings = self.data_manager.get_combined_embeddings(
            source_service, target_service, 'method_only'
        )
        
        # Create a temporary Annoy index for cross-service search
        combined_index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
        source_lookup = {}
        
        # Add source service embeddings to index
        idx = 0
        for method_key in source_labeled:
            if method_key in source_embeddings:
                combined_index.add_item(idx, source_embeddings[method_key])
                source_lookup[idx] = method_key
                idx += 1
        
        combined_index.build(config.ANNOY_N_TREES)
        
        # Predict for target service methods
        for method_key in target_methods:
            if method_key in target_embeddings and method_key not in self.data_manager.method_labels:
                embedding = target_embeddings[method_key]
                
                # Get neighbors from source service
                neighbor_indices = combined_index.get_nns_by_vector(embedding, k)
                
                # Calculate similarities for all neighbors
                all_neighbors = []
                for idx in neighbor_indices:
                    if idx in source_lookup:
                        neighbor_key = source_lookup[idx]
                        neighbor_embedding = source_embeddings[neighbor_key]
                        similarity = cosine_similarity([embedding], [neighbor_embedding])[0][0]
                        all_neighbors.append((neighbor_key, similarity))
                
                if all_neighbors:
                    # Filter by threshold for high-confidence predictions
                    valid_neighbors = [(nk, sim) for nk, sim in all_neighbors if sim >= threshold]
                    
                    # Use valid neighbors if any, otherwise use all neighbors
                    neighbors_to_use = valid_neighbors if valid_neighbors else all_neighbors
                    
                    label_weights = defaultdict(float)
                    label_counts = defaultdict(int)
                    for neighbor_key, similarity in neighbors_to_use:
                        label = self.data_manager.method_labels[neighbor_key]
                        label_weights[label] += similarity
                        label_counts[label] += 1
                    
                    predicted_label = max(label_weights, key=label_weights.get)
                    total_neighbors = len(neighbors_to_use)
                    confidence = label_counts[predicted_label] / total_neighbors
                    max_similarity = max(sim for _, sim in neighbors_to_use)
                    
                    predictions[method_key[1]] = {
                        'label': predicted_label,
                        'confidence': confidence,
                        'similarity': max_similarity,
                        'source_neighbors': [neighbor_key for neighbor_key, _ in neighbors_to_use],
                        'used_threshold': len(valid_neighbors) > 0
                    }
        
        return predictions
