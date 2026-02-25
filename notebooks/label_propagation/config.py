"""Configuration settings for AWS Label Propagation system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path("../..") # Go up two levels from notebooks/label_propagation to workspace
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
ANNOY_INDEXES_DIR = EMBEDDINGS_DIR / "annoy_indexes"
LABELS_FILE = Path(".") / "labels.csv"
ANIMATIONS_DIR = Path(".") / "animations"
HISTORY_FILE = ANIMATIONS_DIR / "history.json"
PREDICTIONS_DIR = Path(".") / "predictions"
SUMMARIES_DIR = Path(".") / "summaries"

# Output files
WITHIN_SERVICE_PREDICTIONS_FILE = PREDICTIONS_DIR / "within_service_predictions.json"
GROUP_CROSS_SERVICE_PRELABELED_FILE = PREDICTIONS_DIR / "group_cross_service_prelabeled_only.json"
GROUP_CROSS_SERVICE_ENHANCED_FILE = PREDICTIONS_DIR / "group_cross_service_enhanced.json"
ALL_TO_ALL_CROSS_SERVICE_PREDICTIONS_FILE = PREDICTIONS_DIR / "all_to_all_cross_service_predictions.json"
CROSS_SERVICE_COMPARISON_FILE = SUMMARIES_DIR / "cross_service_comparison.json"
GROUP_COMPARISON_FILE = SUMMARIES_DIR / "group_methods_comparison.json"
PROPAGATION_SUMMARY_FILE = SUMMARIES_DIR / "propagation_summary.json"
VISUALIZATION_SUMMARY_FILE = SUMMARIES_DIR / "visualization_summary.json"

# Model settings
EMBEDDING_DIM = 2560  # Qwen3-4B default dimension

# Propagation parameters
DEFAULT_K_NEIGHBORS = 15
DEFAULT_SERVICE_THRESHOLD = 0.9
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_THRESHOLD = 0.5

# Evaluation parameters
TEST_SIZE = 0.3
K_VALUES_TO_TEST = [5, 7, 9, 11, 13, 15, 17]  # k values for KNN

# Services
LABELED_SERVICES = ['S3', 'DynamoDB', 'Lambda', 'EC2', 'IAM', 'SSM', 'SQS', 'SNS']
SERVICE_TO_ANIMATE = "EC2"  # Default service for animation

# Cross-Service Groups Configuration
CROSS_SERVICE_GROUPS = {
    'storage_services': {
        'core_services': ['S3'],
        'target_services': ['EFS', 'FSx', 'Glacier', 'EBS'],
        'description': 'Storage and file system services'
    },
    'database_services': {
        'core_services': ['DynamoDB'],
        'target_services': ['RDS', 'Neptune', 'DocumentDB', 'SimpleDB', 'ElastiCache'],
        'description': 'Database and data storage services'
    },
    'compute_services': {
        'core_services': ['Lambda'],
        'target_services': ['ECS', 'AppRunner'], 
        'description': 'Compute and container services'
    },
    'messaging_services': {
        'core_services': ['SQS', 'SNS'],
        'target_services': ['EventBridge', 'Kinesis', 'Pinpoint', 'SES', 'SESV2'],
        'description': 'Messaging and event services'
    },
    'security_management': {
        'core_services': ['IAM'],
        'target_services': ['STS', 'CognitoIdentity', 'SSO'],
        'description': 'Identity, access management and system management services'
    },
    'infrastructure_services': {
        'core_services': ['EC2'],
        'target_services': ['EKS', 'Lightsail'],
        'description': 'Infrastructure and compute instance services'
    }
}

# Annoy index parameters
ANNOY_N_TREES = 10
ANNOY_METRIC = 'angular'  # angular = cosine distance