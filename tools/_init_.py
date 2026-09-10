"""CrewAI-compatible tools for the customer support agents.

Each tool is a CrewAI BaseTool with typed Pydantic arguments and deterministic
behaviour. The pipeline calls these tools directly (via ``_run``) to avoid an
extra LLM round-trip, while they remain usable as CrewAI agent tools.
"""