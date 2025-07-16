# 🔬 Automatic API Analysis and Classification through Deep Learning

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Master's Thesis Research Project**  
> Universidad Politécnica de Madrid (UPM)  
> Master of Science in Machine Learning and Big Data

## 📋 Project Overview

This repository contains the complete codebase for a master's thesis focused on developing automated systems for API analysis and classification using deep learning techniques. The project involves scraping, parsing, and analyzing AWS API documentation to extract structured information for machine learning applications.

**Author:** Alejandro Pardo Bascuñana  
**Supervisor:** Jorge Blasco Alís  
**Institution:** Universidad Politécnica de Madrid (UPM)

## 🚀 Features

- **Automated AWS API Documentation Parsing**: Comprehensive scraping and parsing of AWS API documentation
- **Structured Data Extraction**: Extract method signatures, parameters, and return types
- **Machine Learning Pipeline**: Classification and analysis of API patterns
- **Robust Error Handling**: Checkpoint system for resumable processing
- **Data Visualization**: Statistical analysis and visualization of API patterns

## 📁 Project Structure

```
TFM-APB-MAADM/
├── 📂 parsing/                    # Core parsing and scraping modules
│   ├── 📂 utils/                 # Utility functions and configurations
│   │   ├── config.py             # Configuration and logging setup
│   │   ├── checkpoint_manager.py # Progress tracking and resumption
│   │   ├── text_cleaner.py       # Text cleaning utilities
│   │   └── timeout.py            # Timeout utilities
│   ├── 📂 parsers/               # Parser modules
│   │   ├── __init__.py           # Package initialization
│   │   ├── method_parser.py      # Individual method parsing logic
│   │   ├── service_parser.py     # Service documentation parsing
│   │   └── service_url_parser.py # Service URL extraction
│   ├── main.py                   # Main entry point for parsing
│   ├── service_processor.py      # Service processing coordination
│   ├── checkpoint.json           # Progress checkpoint data
│   └── README.md                 # Parsing module documentation
├── 📂 docs/                      # Documentation and parsed data
│   ├── aws_api_urls.txt          # AWS API URLs
│   ├── 📂 methods/               # Extracted method information by service
│   └── 📂 services/              # Service-specific data
├── 📊 statistics/                # Statistical analysis
│   ├── statistics.ipynb          # Statistical analysis notebook
│   └── unique_action_verbs.txt   # Unique action verbs found in APIs
├── 📂 embeddings/                # Embedding generation and analysis
│   └── embeddings.ipynb          # Embedding generation notebook
├── 📋 requirements.txt           # Python dependencies
├── 📄 LICENSE                    # GPL v3 License
└── 📄 README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/TFM-APB-MAADM.git
   cd TFM-APB-MAADM
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### AWS API Documentation Parsing

To start parsing AWS API documentation:

```bash
cd parsing
python main.py
```

The parser will:
- Scrape AWS API documentation
- Extract structured information about methods and parameters
- Save progress with checkpoint system for resumable processing
- Generate JSON files with parsed data

### Data Analysis

Open the Jupyter notebooks for analysis:

```bash
# Statistical analysis
jupyter notebook statistics/statistics.ipynb

# Embedding generation and analysis
jupyter notebook embeddings/embeddings.ipynb
```

## 📊 Data Processing Pipeline

1. **Web Scraping**: Automated extraction of AWS API documentation
2. **Data Cleaning**: Text preprocessing and normalization
3. **Feature Extraction**: Method signatures, parameters, and metadata
4. **Machine Learning**: Classification and pattern analysis
5. **Visualization**: Statistical insights and data exploration

## 🔧 Configuration

The parsing system can be configured through `parsing/utils/config.py`:

- **Logging levels**: Adjust verbosity of output
- **Checkpoint settings**: Configure resumable processing
- **Output formats**: Customize data export formats
- **Timeout settings**: Configure request timeouts

## 📈 Results

This research contributes to:
- **Automated API Documentation Analysis**: Scalable parsing of large API documentation sets
- **Machine Learning for API Classification**: Novel approaches to API pattern recognition
- **Data-Driven API Insights**: Statistical analysis of API design patterns

## 🤝 Contributing

As this is an academic research project, contributions are welcome for:
- Bug fixes and improvements
- Additional analysis techniques
- Documentation enhancements
- Code optimization

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This work is part of a Master's thesis in the Machine Learning and Big Data program at Universidad Politécnica de Madrid. The research focuses on applying deep learning techniques to automated API analysis and classification.

## 📞 Contact

**Alejandro Pardo Bascuñana**  
Master's Student - Machine Learning and Big Data  
Universidad Politécnica de Madrid (UPM)

For questions about this research, please open an issue in this repository.

---

*This research was conducted as part of the Master of Science in Machine Learning and Big Data at Universidad Politécnica de Madrid under the supervision of Jorge Blasco Alís.*
