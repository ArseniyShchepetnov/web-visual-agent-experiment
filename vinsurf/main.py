"""Entrypoint."""

import phoenix as px
from phoenix.otel import register
from qdrant_client import QdrantClient
from seleniumbase import SB

# Start Phoenix
px.launch_app()

# Set up tracing
tracer_provider = register(project_name="qdrant-app")
tracer = tracer_provider.get_tracer(__name__)

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

with SB(test=True, uc=True) as sb:
    sb.open("https://google.com")
