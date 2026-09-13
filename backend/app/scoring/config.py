"""Central scoring configuration for PromptLens (Step 13).

Defines signal weights and dynamic renormalization logic for handling
available vs. unavailable signals (such as a missing local LLM).
"""
from dataclasses import dataclass, field


SCORING_VERSION = "0.1.0"

# Initial base configured weights summing to 1.0 (100%)
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "rules": 0.25,
    "nlp": 0.15,
    "quality_ml": 0.20,
    "llm": 0.40,
}


@dataclass(frozen=True)
class ScoringConfig:
    version: str = SCORING_VERSION
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGNAL_WEIGHTS))

    def calculate_effective_weights(self, available_signals: set[str]) -> dict[str, float]:
        """Dynamically renormalize weights over available signals.

        If a signal is unavailable (such as the LLM in Step 13), it is not
        treated as zero score. Instead, its weight is redistributed proportionally
        among the signals that are actually present.

        Args:
            available_signals: Set of signal names that are active (e.g. {"rules", "nlp", "quality_ml"}).

        Returns:
            Dict mapping each configured signal name to its effective normalized weight.
            Unavailable signals receive an effective weight of 0.0.
            Available signals sum to 1.0 (unless no signals are available).
        """
        valid_available = {sig for sig in available_signals if sig in self.weights}
        total_available_weight = sum(self.weights[sig] for sig in valid_available)

        if total_available_weight <= 0.0:
            return {sig: 0.0 for sig in self.weights}

        effective: dict[str, float] = {}
        for sig, configured in self.weights.items():
            if sig in valid_available:
                effective[sig] = round(configured / total_available_weight, 6)
            else:
                effective[sig] = 0.0

        # Adjust any tiny floating point rounding on the last available item so the sum is exactly 1.0
        if valid_available:
            total_sum = sum(effective[sig] for sig in valid_available)
            diff = 1.0 - total_sum
            if abs(diff) > 1e-7:
                last_sig = sorted(valid_available)[-1]
                effective[last_sig] = round(effective[last_sig] + diff, 6)

        return effective


DEFAULT_SCORING_CONFIG = ScoringConfig()
