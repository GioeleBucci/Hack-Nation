"""Agent-facing tools, referenced by ``callable:`` in agents/lab_director.yaml.

Each tool takes JSON-like keyword arguments and returns a JSON-serializable dict. Tools never
raise: failures come back as ``{"ok": false, "error": "..."}`` so the calling agent can read
the message and correct itself.
"""
