"""pet-eval: evaluation pipeline for the Train-Pet-Pipeline project.

Provides metric plugins and evaluator plugins (VLM / audio / quantized VLM
plus rule-based cross-modal fusion) registered into the shared
``pet_infra.registry`` via the ``pet_infra.plugins`` entry point.
"""

__version__ = "2.5.1"
