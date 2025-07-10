"""
Parsers package for AWS API documentation processing.

This package contains modules for parsing AWS service documentation:
- method_parser: Parses individual AWS method documentation
- service_url_parser: Extracts service URLs from AWS documentation (formerly parse_available_services)
- service_parser: Parses service documentation files (formerly parse_each_service)
"""

from .method_parser import MethodParser

__all__ = ['MethodParser']
