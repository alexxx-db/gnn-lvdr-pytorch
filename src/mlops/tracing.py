"""
MLflow tracing for the agentic pipeline.

Adds observability to agent tool calls and recommendation explanations.
"""
import mlflow
from functools import wraps


def traced(name: str = None):
    """
    Decorator that wraps a function with MLflow tracing.

    Usage:
        @traced("explain_recommendation")
        def explain(patient_id, care_site_id):
            ...
    """
    def decorator(func):
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with mlflow.start_span(name=span_name) as span:
                span.set_inputs({"args": str(args)[:500], "kwargs": str(kwargs)[:500]})
                try:
                    result = func(*args, **kwargs)
                    span.set_outputs({"result": str(result)[:1000]})
                    return result
                except Exception as e:
                    span.set_status("ERROR")
                    span.set_outputs({"error": str(e)})
                    raise

        return wrapper
    return decorator
