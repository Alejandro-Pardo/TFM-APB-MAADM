"""Data management for embeddings, labels, and Annoy indexes."""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from annoy import AnnoyIndex
import config


class DataManager:
    def __init__(self):
        """Initialize the data manager."""
        self.manual_labels = self._load_manual_labels()
        self.method_embeddings = {}
        self.method_labels = {}
        self.service_methods = defaultdict(list)
        self.annoy_indexes = {}
        self.method_lookups = {}
        
        print(f"🏷️ Loaded {len(self.manual_labels)} manual labels")
    
    def _load_manual_labels(self) -> Dict[Tuple[str, str], str]:
        """Load manual labels from CSV file."""
        df = pd.read_csv(config.LABELS_FILE, sep=';')
        labels = {}
        
        current_service = None
        for _, row in df.iterrows():
            service = row['API Service'].strip() if pd.notna(row['API Service']) and row['API Service'].strip() else current_service
            if service and service != current_service:
                current_service = service
            
            method = row['Methods'].strip() if pd.notna(row['Methods']) else None
            label = row['Sink/Source'].strip() if pd.notna(row['Sink/Source']) else None
            
            if method and label and current_service:
                # Normalize labels
                if label.lower() in ['source', 'source / x']:
                    normalized_label = 'source'
                elif label.lower() in ['sink']:
                    normalized_label = 'sink'
                else:
                    normalized_label = 'none'
                
                labels[(current_service.lower(), method)] = normalized_label
        
        return labels
    
    def load_method_embeddings(self, services: List[str], embedding_format: str) -> None:
        """
        Load method embeddings for specified services.
        
        Args:
            services: List of service names to load
            embedding_format: Format of embeddings to load
        """
        methods_dir = config.EMBEDDINGS_DIR / "methods"
        
        if services is None:
            services = [d.name for d in methods_dir.iterdir() if d.is_dir()]
        
        print(f"📂 Loading embeddings for services: {services}")
        print(f"📊 Using embedding format: {embedding_format}")
        
        for service in services:
            # Find the actual service directory (case insensitive)
            service_dir = None
            for dir_path in methods_dir.iterdir():
                if dir_path.is_dir() and dir_path.name.lower() == service.lower():
                    service_dir = dir_path
                    break
            
            if not service_dir or not service_dir.exists():
                print(f"⚠️ Service directory not found: {service}")
                continue
            
            method_files = list(service_dir.glob("*.json"))
            print(f"📁 {service_dir.name}: {len(method_files)} methods")
            
            for method_file in method_files:
                with open(method_file, 'r') as f:
                    method_data = json.load(f)
                    
                    service_name = method_data['service_name'].lower()
                    method_name = method_data['method_name']
                    
                    if embedding_format in method_data.get('embeddings', {}):
                        embedding = np.array(method_data['embeddings'][embedding_format])
                        
                        # Validate dimension consistency
                        if len(embedding) != config.EMBEDDING_DIM:
                            print(f"⚠️ Warning: Embedding dimension mismatch for {service_name}.{method_name}: "
                                  f"expected {config.EMBEDDING_DIM}, got {len(embedding)}")
                            continue
                        
                        key = (service_name, method_name)
                        
                        self.method_embeddings[key] = embedding
                        self.service_methods[service_name].append(method_name)
                        
                        # Add label if we have it manually labeled
                        if key in self.manual_labels:
                            self.method_labels[key] = self.manual_labels[key]
        
        print(f"✅ Loaded {len(self.method_embeddings)} method embeddings")
        print(f"🏷️ Found {len(self.method_labels)} labeled methods")
    
    def build_service_index(self, service: str) -> None:
        """Build Annoy index for a specific service."""
        service = service.lower()
        service_methods = [(service, method) for method in self.service_methods[service]]
        
        # Create Annoy index
        index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
        method_lookup = {}
        
        # Add embeddings to index
        idx = 0
        for method_key in service_methods:
            if method_key in self.method_embeddings:
                index.add_item(idx, self.method_embeddings[method_key])
                method_lookup[idx] = method_key
                idx += 1
        
        # Build index
        index.build(config.ANNOY_N_TREES)
        
        # Store index and lookup
        self.annoy_indexes[service] = index
        self.method_lookups[service] = method_lookup
        
        print(f"   🔧 Built Annoy index for {service}: {len(method_lookup)} methods")
    
    def ensure_service_index(self, service: str) -> None:
        """Ensure Annoy index exists for a service."""
        service = service.lower()
        if service not in self.annoy_indexes:
            self.build_service_index(service)
    
    def save_indexes(self) -> None:
        """Save Annoy indexes for fast loading."""        
        for service, index in self.annoy_indexes.items():
            # Create service-specific directory
            service_dir = config.ANNOY_INDEXES_DIR / service
            service_dir.mkdir(exist_ok=True)
            
            # Save Annoy index
            index_path = service_dir / f"{service}_index.ann"
            index.save(str(index_path))
            
            # Save lookup table
            lookup_path = service_dir / f"{service}_lookup.pkl"
            with open(lookup_path, 'wb') as f:
                pickle.dump(self.method_lookups[service], f)
        
        print(f"💾 Annoy indexes saved to: {config.ANNOY_INDEXES_DIR}")
    
    def load_indexes(self) -> None:
        """Load pre-built Annoy indexes."""
        if not config.ANNOY_INDEXES_DIR.exists():
            print(f"⚠️ Index directory not found: {config.ANNOY_INDEXES_DIR}")
            return
        
        for service in self.service_methods.keys():
            service_dir = config.ANNOY_INDEXES_DIR / service
            
            if not service_dir.exists():
                continue
                
            index_path = service_dir / f"{service}_index.ann"
            lookup_path = service_dir / f"{service}_lookup.pkl"
            
            if index_path.exists() and lookup_path.exists():
                # Load Annoy index
                index = AnnoyIndex(config.EMBEDDING_DIM, config.ANNOY_METRIC)
                index.load(str(index_path))
                
                # Load lookup table
                with open(lookup_path, 'rb') as f:
                    lookup = pickle.load(f)
                
                self.annoy_indexes[service] = index
                self.method_lookups[service] = lookup
                
                print(f"📂 Loaded Annoy index for {service}: {len(lookup)} methods")
    
    def save_predictions(self, predictions: Dict[str, Dict[str, str]], 
                        output_file: Path, metadata: Dict = None) -> None:
        """Save predictions to JSON file."""
        # Convert to serializable format
        serializable_predictions = {}
        for service, service_predictions in predictions.items():
            serializable_predictions[service] = {}
            for method, pred_data in service_predictions.items():
                if isinstance(pred_data, dict):
                    # Convert numpy types to native Python types
                    serializable_pred = {}
                    for key, value in pred_data.items():
                        if isinstance(value, np.floating):
                            serializable_pred[key] = float(value)
                        elif isinstance(value, list):
                            serializable_pred[key] = [str(item) for item in value]
                        else:
                            serializable_pred[key] = value
                    serializable_predictions[service][method] = serializable_pred
                else:
                    serializable_predictions[service][method] = pred_data
        
        # Add metadata if provided
        if metadata:
            output_data = {
                'metadata': metadata,
                'predictions': serializable_predictions
            }
        else:
            output_data = serializable_predictions
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Predictions saved to: {output_file}")
    
    def flatten_group_predictions(self, group_predictions: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
        """
        Flatten group-based predictions to service-level format for easier processing.
        
        Args:
            group_predictions: {group_name: {service: {method: prediction}}}
            
        Returns:
            {service: {method: prediction}} - flattened format
        """
        flattened = {}
        
        for group_name, group_data in group_predictions.items():
            for service, service_predictions in group_data.items():
                if service not in flattened:
                    flattened[service] = {}
                
                # Add group information to each prediction
                for method, pred_data in service_predictions.items():
                    if isinstance(pred_data, dict):
                        pred_data_copy = pred_data.copy()
                        pred_data_copy['source_group'] = group_name
                        flattened[service][method] = pred_data_copy
                    else:
                        flattened[service][method] = {
                            'label': pred_data,
                            'source_group': group_name
                        }
        
        return flattened
    
    def get_service_embeddings(self, service: str, embedding_format: str) -> Dict:
        """
        Get embeddings for a single service using the specified format.
        
        Args:
            service: Service name
            embedding_format: Format of embeddings to load ('method_only', 'with_params', etc.)
            
        Returns:
            Dictionary of {(service, method): embedding_vector}
        """
        service = service.lower()
        service_embeddings = {}
        
        if embedding_format == 'method_only':
            # Load method_only embeddings from disk
            methods_dir = config.EMBEDDINGS_DIR / "methods"
            
            # Find the actual service directory (case insensitive)
            service_dir = None
            for dir_path in methods_dir.iterdir():
                if dir_path.is_dir() and dir_path.name.lower() == service.lower():
                    service_dir = dir_path
                    break
            
            if service_dir and service_dir.exists():
                for method_file in service_dir.glob("*.json"):
                    try:
                        with open(method_file, 'r') as f:
                            method_data = json.load(f)
                            service_name = method_data['service_name'].lower()
                            method_name = method_data['method_name']
                            
                            if ('method_only' in method_data.get('embeddings', {}) and 
                                service_name == service):
                                embedding = np.array(method_data['embeddings']['method_only'])
                                if len(embedding) == config.EMBEDDING_DIM:
                                    key = (service_name, method_name)
                                    service_embeddings[key] = embedding
                    except Exception as e:
                        print(f"⚠️ Error loading {method_file}: {e}")
        else:
            # Use existing embeddings in memory
            service_embeddings = {k: v for k, v in self.method_embeddings.items() 
                                if k[0] == service}
        
        return service_embeddings
    
    def get_combined_embeddings(self, service1: str, service2: str, 
                               embedding_format: str) -> Tuple[Dict, Dict]:
        """
        Get embeddings for two services using different formats.
        Used for cross-service propagation.
        
        Returns:
            Tuple of (service1_embeddings, service2_embeddings)
        """
        # For cross-service, we need to reload with 'method_only' format
        if embedding_format == 'method_only':
            # Temporarily store current embeddings
            temp_embeddings = {}
            
            # Load embeddings for both services with method_only format
            methods_dir = config.EMBEDDINGS_DIR / "methods"
            
            for service in [service1, service2]:
                # Find the actual service directory (case insensitive)
                service_dir = None
                for dir_path in methods_dir.iterdir():
                    if dir_path.is_dir() and dir_path.name.lower() == service.lower():
                        service_dir = dir_path
                        break
                
                if not service_dir or not service_dir.exists():
                    continue
                
                for method_file in service_dir.glob("*.json"):
                    with open(method_file, 'r') as f:
                        method_data = json.load(f)
                        service_name = method_data['service_name'].lower()
                        method_name = method_data['method_name']
                        
                        if 'method_only' in method_data.get('embeddings', {}):
                            embedding = np.array(method_data['embeddings']['method_only'])
                            if len(embedding) == config.EMBEDDING_DIM:
                                key = (service_name, method_name)
                                temp_embeddings[key] = embedding
            
            # Filter for each service
            service1_embeddings = {k: v for k, v in temp_embeddings.items() if k[0] == service1.lower()}
            service2_embeddings = {k: v for k, v in temp_embeddings.items() if k[0] == service2.lower()}
            
            return service1_embeddings, service2_embeddings
        else:
            # Use existing embeddings
            service1_embeddings = {k: v for k, v in self.method_embeddings.items() if k[0] == service1.lower()}
            service2_embeddings = {k: v for k, v in self.method_embeddings.items() if k[0] == service2.lower()}
            return service1_embeddings, service2_embeddings
