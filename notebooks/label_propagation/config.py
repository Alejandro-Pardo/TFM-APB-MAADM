"""Configuration settings for AWS Label Propagation system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path("../..") # Go up two levels from notebooks/label_propagation to workspace
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
ANNOY_INDEXES_DIR = EMBEDDINGS_DIR / "annoy_indexes"
LABELS_FILE = Path(".") / "labels.csv"

# Output files
WITHIN_SERVICE_PREDICTIONS_FILE = Path(".") / "within_service_predictions.json"
CROSS_SERVICE_PREDICTIONS_FILE = Path(".") / "cross_service_predictions.json"
PROPAGATION_SUMMARY_FILE = Path(".") / "propagation_summary.json"
VISUALIZATION_SUMMARY_FILE = Path(".") / "visualization_summary.json"

# Model settings
EMBEDDING_DIM = 1024  # Qwen3-0.6B default dimension

# Propagation parameters
DEFAULT_K_NEIGHBORS = 5
DEFAULT_WITHIN_SERVICE_THRESHOLD = 0.9
DEFAULT_CROSS_SERVICE_THRESHOLD = 0.8
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_THRESHOLD = 0.1

# Evaluation parameters
TEST_SIZE = 0.3
K_VALUES_TO_TEST = [3, 5, 7, 10]
# THRESHOLDS_TO_TEST removed - using adaptive thresholding instead

# Services
LABELED_SERVICES = ['S3', 'DynamoDB', 'Lambda', 'EC2', 'IAM', 'SSM', 'SQS', 'SNS']

# Similar services mapping for cross-service propagation
SIMILAR_SERVICES = {
    's3': ['efs', 'fsx', 'backup'],  # Storage services
    'dynamodb': ['rds', 'elasticache', 'neptune', 'documentdb'],  # Database services  
    'lambda': ['ecs', 'batch', 'stepfunctions'],  # Compute services
    'sqs': ['sns', 'eventbridge', 'kinesis', 'mq'],  # Messaging services
    'iam': ['sts', 'ssm', 'cloudwatch'],  # Management services
    'ec2': ['autoscaling', 'elb', 'ecs'],  # Infrastructure services
    'sns': ['sqs', 'eventbridge', 'pinpoint'],  # Messaging services
    'ssm': ['iam', 'cloudwatch', 'config']  # Management services
}

# Cross-service test pairs
CROSS_SERVICE_TEST_PAIRS = [
    ('s3', 'efs'),
    ('dynamodb', 'rds'), 
    ('sqs', 'eventbridge'),
    ('iam', 'sts')
]

# Annoy index parameters
ANNOY_N_TREES = 10
ANNOY_METRIC = 'angular'  # angular = cosine distance
