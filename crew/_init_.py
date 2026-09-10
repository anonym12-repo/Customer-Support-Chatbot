"""CrewAI agents for the customer support system.

This package is the single source of truth for CrewAI: each specialist agent and
its crew live here. The pipeline pre-gathers deterministic context and asks the
relevant agent to generate a single grounded answer (one LLM call per request).
"""