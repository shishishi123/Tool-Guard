import agentdojo.attacks.baseline_attacks  # -- needed to register the attacks
import agentdojo.attacks.dos_attacks  # -- needed to register the attacks
import agentdojo.attacks.important_instructions_attacks  # noqa: F401  -- needed to register the attacks
import agentdojo.attacks.tool_description_poisoning  # noqa: F401 -- needed to register the attacks
from agentdojo.attacks.attack_registry import load_attack, register_attack
from agentdojo.attacks.base_attacks import BaseAttack, FixedJailbreakAttack
from agentdojo.attacks.tool_description_poisoning import (
    # Attack classes
    AdaptiveSplitReplanAttack,
    AlignmentAdaptiveAttack,
    CombinedAdaptiveAttack,
    PairAdaptiveAttack,
    TapAdaptiveAttack,
    SuspicionAdaptiveAttack,
    AttackerToolInjector,
    ToolDescriptionPoisoner,
    ToolDescriptionPoisoningAttack,
    ToolDescriptionPoisoningElement,
    # Strategies
    adaptive_authority_strategy,
    adaptive_append_strategy,
    adaptive_subtle_strategy,
    adaptive_prepend_strategy,
    adaptive_split_replan_strategy,
    alignment_adaptive_strategy,
    append_instruction_strategy,
    authority_injection_strategy,
    combined_adaptive_strategy,
    pair_adaptive_strategy,
    tap_adaptive_strategy,
    prepend_instruction_strategy,
    replace_with_malicious_strategy,
    subtle_redirect_strategy,
    suspicion_adaptive_strategy,
)

__all__ = [
    # Attack classes
    "AdaptiveSplitReplanAttack",
    "AlignmentAdaptiveAttack",
    "CombinedAdaptiveAttack",
    "PairAdaptiveAttack",
    "TapAdaptiveAttack",
    "SuspicionAdaptiveAttack",
    "AttackerToolInjector",
    "BaseAttack",
    "FixedJailbreakAttack",
    "ToolDescriptionPoisoner",
    "ToolDescriptionPoisoningAttack",
    "ToolDescriptionPoisoningElement",
    # Strategies
    "adaptive_authority_strategy",
    "adaptive_append_strategy",
    "adaptive_subtle_strategy",
    "adaptive_prepend_strategy",
    "adaptive_split_replan_strategy",
    "alignment_adaptive_strategy",
    "append_instruction_strategy",
    "authority_injection_strategy",
    "combined_adaptive_strategy",
    "pair_adaptive_strategy",
    "tap_adaptive_strategy",
    "load_attack",
    "prepend_instruction_strategy",
    "register_attack",
    "replace_with_malicious_strategy",
    "subtle_redirect_strategy",
    "suspicion_adaptive_strategy",
]
