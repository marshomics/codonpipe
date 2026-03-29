"""CodonPipe analysis modules.

Error handling convention
-------------------------
Modules in this package follow a two-tier error model:

* **Return empty / None** for "nothing to do" conditions that the pipeline
  can gracefully skip.  Examples: no ribosomal proteins found, no KO hits,
  too few genes for GMM clustering.

* **Raise an exception** (``RuntimeError``, ``FileNotFoundError``,
  ``ValueError``) for genuine failures that indicate a broken dependency,
  corrupt input, or missing required tool.  Examples: external tool not
  installed, output file not produced, COG result table missing required
  columns.

The pipeline orchestrator (``pipeline.py``) wraps each step in a
``try / except`` block, logs the error, and continues to the next step.
This keeps individual module contracts clean while letting the pipeline
degrade gracefully.
"""
