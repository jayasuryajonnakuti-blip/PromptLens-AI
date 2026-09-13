"""PromptLens AI Optimizer package (Step 16).

Generates an improved prompt based on the original prompt and a Step 15
AnalyzerAnalysis result.  Supports four optimization modes:

    balanced   — Improve clarity, structure and completeness proportionally
    analytical — Emphasise logical structure, precision and explicit constraints
    creative   — Allow richer expression while retaining the original intent
    expert     — Target a domain-expert audience with precise technical language

The optimizer NEVER:
  - Invent facts or requirements not present in the original prompt
  - Expand scope beyond what the analyzer identified
  - Silently fall back to mock output in production
"""
