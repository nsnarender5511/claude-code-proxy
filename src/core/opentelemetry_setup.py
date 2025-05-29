from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPGrpcSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import litellm
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)


def setup_opentelemetry(app):
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.info(
            "OpenTelemetry export endpoint not configured. Skipping OpenTelemetry setup."
        )
        return
    resource = Resource(attributes={"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter = OTLPGrpcSpanExporter(settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(
        set_logging_format=True,
        log_level=logging.getLevelName(settings.LOG_LEVEL.upper()),
    )
