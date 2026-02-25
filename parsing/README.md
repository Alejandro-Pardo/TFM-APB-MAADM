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
- **`service_processor.py`** - Service-level processing coordination
- **`utils/config.py`** - Configuration management and logging setup
- **`utils/checkpoint_manager.py`** - Progress tracking and resumable processing
- **`utils/text_cleaner.py`** - Text cleaning and formatting utilities (DescriptionCleaner class)
- **`utils/timeout.py`** - Timeout decorators and timeout exception handling
- **`parsers/method_parser.py`** - Individual method documentation parsing
- **`parsers/service_parser.py`** - Service documentation parsing
- **`parsers/service_url_parser.py`** - Service URL extraction

### Module Dependencies

```
main.py
├── utils/config.py
├── utils/checkpoint_manager.py
└── service_processor.py
    └── parsers/
        ├── method_parser.py
        │   ├── utils/text_cleaner.py
        │   ├── utils/timeout.py
        │   └── utils/config.py
        ├── service_parser.py
        │   └── utils/text_cleaner.py
        └── service_url_parser.py
```

## Features

### Robust Processing
- **Timeout Protection**: Function-level timeouts using threading to prevent hanging
- **HTTP Retry Logic**: Built-in retry mechanisms for network requests
- **Progress Tracking**: Checkpoint system with resumable processing from interruptions
- **Error Handling**: Comprehensive logging with clean console output and detailed file logs

### Data Extraction
- **Method Information**: Complete method signatures, descriptions, and documentation URLs
- **Parameter Analysis**: Hierarchical parameter structures with types, requirements, and descriptions
- **Return Structures**: Detailed return value documentation with nested item support
- **Text Cleaning**: Advanced text processing using DescriptionCleaner for encoding and duplicate removal

### Scalability
- **Modular Design**: Clean separation with utils/, parsers/, and core processing modules
- **Memory Efficient**: Individual service processing to minimize memory footprint
- **Rate Limiting**: Configurable delays between requests and services
- **Graceful Interruption**: Keyboard interrupt handling with progress preservation

## Installation

### Requirements

The project has a comprehensive `requirements.txt` file located in the project root. The main parsing dependencies include:

```
requests>=2.31.0
beautifulsoup4>=4.12.0
tqdm>=4.65.0
urllib3>=2.0.0
```

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

**Note**: The project includes many additional dependencies for machine learning and data processing components beyond the parsing module.

### Setup

1. Clone or download the project files
2. Ensure the following directory structure exists:
   ```
   parsing/
   ├── main.py
   ├── service_processor.py
   ├── checkpoint.json         # Created after first run
   ├── parser.log             # Created after first run
   ├── utils/
   │   ├── __init__.py
   │   ├── config.py
   │   ├── checkpoint_manager.py
   │   ├── text_cleaner.py
   │   └── timeout.py
   ├── parsers/
   │   ├── __init__.py
   │   ├── method_parser.py
   │   ├── service_parser.py
   │   └── service_url_parser.py
   └── __pycache__/          # Created during execution
   ```

## Usage

### Basic Usage

Run the main script to process all AWS services:

```bash
python main.py
```

### Configuration

Modify `utils/config.py` to adjust default settings:

```python
DEFAULT_CONFIG = {
    'services_folder': os.path.join(_project_root, "docs", "services"),
    'output_folder': os.path.join(_project_root, "docs", "methods"),
    'checkpoint_file': os.path.join(_parsing_dir, "checkpoint.json"),
    'request_timeout': 50,                     # HTTP request timeout (seconds)
    'processing_timeout': 300,                 # Service processing timeout (seconds)
    'sleep_between_requests': 1.0,             # Delay between method requests (seconds)
    'sleep_between_services': 0.5              # Delay between service processing (seconds)
}
```

**Note**: The configuration uses absolute paths calculated from the current file location for better reliability.

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

### Generated Files During Execution

The parser creates several files during execution:

- **`checkpoint.json`** - Progress tracking file for resumable processing
- **`parser.log`** - Detailed log file with timestamps and full error traces
- **`../docs/methods/[ServiceName]/`** - Individual method JSON files organized by service
- **`../docs/methods/[ServiceName]/_summary.json`** - Service-level summary files
- **`../docs/methods/all_services_summary.json`** - Complete processing overview

## Error Handling

### Failure Tracking

Failed operations are logged to specific files:
- `parser.log` - Detailed application logs with timestamps
- Output to `../docs/methods/` - Individual method JSON files and service summaries
- Console output with progress bars and status updates

### Common Issues and Solutions

1. **Network Timeouts**: Increase `request_timeout` in configuration
2. **Rate Limiting**: Increase `sleep_between_requests` value
3. **Memory Issues**: Process services individually (default behavior)
4. **Encoding Problems**: The `DescriptionCleaner` handles most common issues

## Development

### Adding New Features

1. **New Extractors**: Add methods to `MethodParser` class in `parsers/method_parser.py`
2. **Data Cleaning**: Extend `DescriptionCleaner` class in `utils/text_cleaner.py`
3. **Processing Logic**: Modify `ServiceProcessor` class in `service_processor.py`
4. **Configuration**: Add new settings to `utils/config.py`
5. **Timeout Handling**: Modify timeout decorators in `utils/timeout.py`

### Testing

Test individual components:

```python
# Test single method parsing
from parsers.method_parser import MethodParser
parser = MethodParser()
result = parser.scrape_method_details(url, method_name)

# Test service processing
from service_processor import ServiceProcessor
from utils.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager()
processor = ServiceProcessor(services_folder, output_folder, checkpoint_manager)
processor.process_service_file(service_file_path)

# Test text cleaning
from utils.text_cleaner import DescriptionCleaner
cleaner = DescriptionCleaner()
cleaned_text = cleaner.clean_description(raw_text)
```

### Logging

The application uses a custom logging system with clean formatting:
- **Console Output**: Clean formatting with emoji prefixes for better readability
- **File Logging**: Detailed timestamped logs in `parser.log`
- **Log Levels**: INFO, WARNING, ERROR, DEBUG (configure in `utils/config.py`)

**Log Format Examples**:
- Console: `[INFO] 📄 Processing service: AccessAnalyzer`
- File: `2025-09-29 10:30:45 - INFO - Processing service: AccessAnalyzer`

### Timeout Handling

The application includes robust timeout protection:
- **Function-level timeouts**: Using `@timeout(seconds)` decorator
- **Request timeouts**: Configurable HTTP request timeouts
- **Service processing timeouts**: 5-minute timeout per service
- **Graceful handling**: TimeoutException for proper error handling

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
