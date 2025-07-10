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
            
            # Check for KeyboardInterrupt while waiting
            import time
            elapsed = 0
            check_interval = 0.1  # Check every 100ms
            
            while t.is_alive() and elapsed < seconds:
                try:
                    time.sleep(check_interval)
                    elapsed += check_interval
                except KeyboardInterrupt:
                    # If Ctrl+C is pressed, let it propagate
                    raise KeyboardInterrupt
            
            # Final join with remaining time
            if t.is_alive():
                t.join(0)  # Don't wait, just check if it finished
                if t.is_alive():
                    # Still running after timeout
                    result[0] = TimeoutException("Function call timed out")
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator
