"""
Caveman Token Compression Skill
Inspired by: https://github.com/JuliusBrussee/caveman

Provides rules for ultra-compressed communication to save tokens and costs
while maintaining technical accuracy.
"""

import re
from typing import Optional

CAVEMAN_RULES = {
    "lite": (
        "Respond professional but extremely tight. "
        "Remove all filler words (just, basically, actually, simply, really). "
        "Remove all pleasantries (sure, certainly, happy to help, of course). "
        "Remove all hedging (I believe, it seems, possibly). "
        "Keep full sentences and articles."
    ),
    "full": (
        "Respond terse like smart caveman. "
        "All technical substance stay. Only fluff die. "
        "Drop: articles (a, an, the), fillers, pleasantries, hedging. "
        "Fragments OK. Short synonyms (fix not implement, big not extensive). "
        "Technical terms exact. Code blocks unchanged. "
        "Pattern: [thing] [action] [reason]. [next step]."
    ),
    "ultra": (
        "Respond with maximum compression (ULTRA CAVEMAN). "
        "Abbreviate whenever possible (DB, auth, config, req, res, fn, impl, UI). "
        "Strip all conjunctions. Use arrows (->) for causality. "
        "One word when one word enough. "
        "Maximum token efficiency is critical."
    )
}

CAVEMAN_TRIGGER_KEYWORDS = [
    "caveman mode",
    "talk like caveman",
    "use caveman",
    "less tokens",
    "be brief",
    "concise mode",
    "terse mode"
]

CAVEMAN_EXIT_KEYWORDS = [
    "stop caveman",
    "normal mode",
    "professional mode",
    "standard mode",
    "exit caveman"
]

def detect_caveman_request(text: str) -> Optional[str]:
    """
    Detects if the user is requesting a change in Caveman mode.
    Returns the intensity level ('lite', 'full', 'ultra') or 'off'.
    Returns None if no change is detected.
    """
    text_lower = text.lower()
    
    # Check for exit
    if any(kw in text_lower for kw in CAVEMAN_EXIT_KEYWORDS):
        return "off"
    
    # Check for activation / intensity change
    if any(kw in text_lower for kw in CAVEMAN_TRIGGER_KEYWORDS) or text_lower.startswith("/caveman"):
        if "ultra" in text_lower:
            return "ultra"
        if "lite" in text_lower:
            return "lite"
        return "full" # default
        
    return None

def get_caveman_system_prompt(intensity: str = "full") -> str:
    """Returns the system prompt instructions for Caveman mode."""
    rules = CAVEMAN_RULES.get(intensity, CAVEMAN_RULES["full"])
    return (
        "\n### CAVEMAN MODE ACTIVATED ###\n"
        f"{rules}\n"
        "STILL ACTIVE: No filler drift. Technical accuracy mandatory. Code blocks stay normal."
    )
