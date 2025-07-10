"""
Timeout utilities for the AWS API parser.

This module contains timeout-related utilities like timeout decorators
and custom exceptions for handling function execution timeouts.
"""

import threading


class TimeoutException(Exception):
    """Raised when a function execution times out."""
    pass


def timeout(seconds):
    """
    Decorator to add timeout functionality to functions using threading.
    
    Args:
        seconds (int): Timeout duration in seconds
    
    Returns:
        function: Decorated function with timeout capability
    
    Raises:
        TimeoutException: If function execution exceeds timeout
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutException("Function call timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(seconds)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator
