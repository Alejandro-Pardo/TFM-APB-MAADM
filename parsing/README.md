# AWS API Documentation Parser

A modular Python application for parsing and extracting structured information from AWS API documentation. This tool scrapes AWS service documentation to create comprehensive JSON datasets containing method details, parameters, and return structures.

## Overview

The parser is designed to process AWS service documentation systematically, extracting:
- Method descriptions and usage information
- Parameter specifications (types, requirements, descriptions)
- Return value structures and types
- Nested parameter and return value hierarchies

## Architecture

The application is structured into several focused modules:

### Core Modules

- **`main.py`** - Main entry point and orchestration
- **`config.py`** - Configuration management and logging setup
- **`utils.py`** - Utility functions including timeout decorators and HTML preprocessing
- **`checkpoint_manager.py`** - Progress tracking and resumable processing
- **`description_cleaner.py`** - Text cleaning and formatting utilities
- **`parsers/method_parser.py`** - Individual method documentation parsing
- **`parsers/service_parser.py`** - Service documentation parsing (formerly parse_each_service.py)
- **`parsers/service_url_parser.py`** - Service URL extraction (formerly parse_available_services.py)
- **`service_processor.py`** - Service-level processing coordination

### Module Dependencies

```
main.py
├── config.py
├── checkpoint_manager.py
└── service_processor.py
    └── parsers/
        ├── method_parser.py
        │   ├── description_cleaner.py
        │   └── utils.py
        ├── service_parser.py
        └── service_url_parser.py
```

## Features

### Robust Processing
- **Timeout Protection**: Prevents hanging on slow or unresponsive pages
- **Retry Logic**: Handles temporary network issues and rate limiting
- **Progress Tracking**: Checkpoint system allows resuming interrupted processing
- **Error Handling**: Comprehensive error logging and failure tracking

### Data Extraction
- **Method Information**: Complete method signatures and descriptions
- **Parameter Analysis**: Detailed parameter specifications including types and requirements
- **Return Structures**: Hierarchical return value documentation
- **Text Cleaning**: Advanced text processing to handle encoding issues and duplicates

### Scalability
- **Modular Design**: Easy to extend and modify individual components
- **Memory Efficient**: Processes services individually to minimize memory usage
- **Rate Limiting**: Respectful scraping with configurable delays

## Installation

### Requirements

Create a `requirements.txt` file with the following dependencies:

```
requests>=2.31.0
beautifulsoup4>=4.12.0
tqdm>=4.65.0
urllib3>=2.0.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone or download the project files
2. Ensure the following directory structure exists:
   ```
   parsing/
   ├── main.py
   ├── config.py
   ├── utils/
   │   ├── checkpoint_manager.py
   │   ├── config.py
   │   ├── text_cleaner.py
   │   └── utils.py
   ├── parsers/
   │   ├── method_parser.py
   │   ├── service_parser.py
   │   └── service_url_parser.py
   ├── service_processor.py
   ├── main.py
   └── requirements.txt
   ```

## Usage

### Basic Usage

Run the main script to process all AWS services:

```bash
python main.py
```

### Configuration

Modify `config.py` to adjust default settings:

```python
DEFAULT_CONFIG = {
    'services_folder': "../docs/services",      # Input folder with service JSON files
    'output_folder': "../docs/methods",        # Output folder for parsed methods
    'checkpoint_file': "checkpoint.json",      # Checkpoint file for progress tracking
    'request_timeout': 50,                     # HTTP request timeout (seconds)
    'processing_timeout': 300,                 # Service processing timeout (seconds)
    'sleep_between_requests': 1.0,             # Delay between method requests (seconds)
    'sleep_between_services': 2.0              # Delay between service processing (seconds)
}
```

### Processing Single Service

To test with a single service, modify `main.py`:

```python
# Uncomment and modify these lines in main.py
service_file = "AccessAnalyzer.json"
service_file_path = os.path.join(services_folder, service_file)
processor.process_service_file(service_file_path)
```

### Resuming Interrupted Processing

The application automatically resumes from the last checkpoint if interrupted. The checkpoint file (`checkpoint.json`) tracks:
- Services that have been completely processed
- Methods that have been processed within each service
- Current service being processed

## Output Structure

### Method Files

Each processed method generates a JSON file with the following structure:

```json
{
  "method_name": "create_analyzer",
  "url": "https://docs.aws.amazon.com/...",
  "description": "Creates an analyzer for the specified type.",
  "parameters": [
    {
      "name": "analyzerName",
      "type": "string",
      "required": true,
      "description": "The name of the analyzer to create.",
      "nested_params": []
    }
  ],
  "return_structure": [
    {
      "name": "arn",
      "type": "string",
      "description": "The ARN of the analyzer that was created.",
      "nested_items": []
    }
  ]
}
```

### Service Summaries

Each service generates a `_summary.json` file:

```json
{
  "service_name": "AccessAnalyzer",
  "url": "https://docs.aws.amazon.com/...",
  "method_count": 15,
  "methods": [
    {
      "name": "create_analyzer",
      "url": "https://docs.aws.amazon.com/...",
      "file_path": "/path/to/create_analyzer.json"
    }
  ]
}
```

### Overall Summary

An `all_services_summary.json` file provides a complete overview:

```json
{
  "total_services": 150,
  "services": [
    {
      "service_name": "AccessAnalyzer",
      "url": "https://docs.aws.amazon.com/...",
      "method_count": 15,
      "methods": [...]
    }
  ]
}
```

## Error Handling

### Failure Tracking

Failed operations are logged to specific files:
- `failed_methods.txt` - Methods that failed to process
- `failed_services.txt` - Services that failed completely
- `parser.log` - Detailed application logs

### Common Issues and Solutions

1. **Network Timeouts**: Increase `request_timeout` in configuration
2. **Rate Limiting**: Increase `sleep_between_requests` value
3. **Memory Issues**: Process services individually (default behavior)
4. **Encoding Problems**: The `DescriptionCleaner` handles most common issues

## Development

### Adding New Features

1. **New Extractors**: Add methods to `MethodParser` class
2. **Data Cleaning**: Extend `DescriptionCleaner` class
3. **Processing Logic**: Modify `ServiceProcessor` class
4. **Configuration**: Add new settings to `config.py`

### Testing

Test individual components:

```python
# Test single method parsing
from parsers.method_parser import MethodParser
parser = MethodParser()
result = parser.scrape_method_details(url, method_name)

# Test service processing
from service_processor import ServiceProcessor
processor = ServiceProcessor(services_folder, output_folder)
processor.process_service_file(service_file_path)
```

### Logging

The application uses structured logging with multiple levels:
- `INFO`: General progress and status
- `WARNING`: Recoverable issues
- `ERROR`: Processing failures
- `DEBUG`: Detailed debugging information (configure in `config.py`)

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Process multiple services in sequence
2. **Caching**: Results are automatically cached per method
3. **Rate Limiting**: Adjust delays based on server response times
4. **Memory Management**: Large services are processed incrementally

### Monitoring Progress

Use the built-in progress indicators:
- Service-level progress bar (tqdm)
- Method-level progress bar per service
- Checkpoint file for overall progress tracking
- Detailed logs for debugging

## License

This project is part of academic research (TFM-APB-MAADM). Please refer to the main project license for usage terms.
