"""
Betting calculations: expected value, Kelly criterion, implied probability.

All functions assume decimal odds (e.g. 1.85 means a $1 bet returns $1.85,
giving a profit of $0.85 on a win).

Key concepts
────────────
• Implied probability  = 1 / decimal_odds
    The probability baked into the odds by the bookmaker (includes their margin).

• Expected value (EV) = p × (odds − 1) − (1 − p)
    The average profit per $1 staked given your true win probability p.
    Positive EV = the bet has long-run value.

• Kelly criterion      = (b·p − q) / b   where b = odds − 1, q = 1 − p
    The fraction of bankroll that maximises the logarithm of wealth growth.
    Full Kelly is theoretically optimal but volatile; half-Kelly is the
    standard conservative recommendation.
"""

from __future__ import annotations


def implied_prob(decimal_odds: float) -> float:
    """Book's implied win probability (overround not removed)."""
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def expected_value(p: float, decimal_odds: float) -> float:
    """
    Expected profit per $1 staked.
    Returns a fraction: 0.05 means +5 cents per dollar bet on average.
    """
    return p * (decimal_odds - 1) - (1 - p)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction of bankroll to bet.
    Clamped to [0, 1] — never bets more than bankroll, never bets negative.
    """
    b = decimal_odds - 1
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, min(f, 1.0))


def kelly_stake(p: float, decimal_odds: float, bankroll: float,
                kelly_mult: float = 0.5) -> float:
    """
    Dollar stake using fractional Kelly.
    kelly_mult=0.5  →  half Kelly (recommended default)
    kelly_mult=1.0  →  full Kelly
    """
    return kelly_fraction(p, decimal_odds) * kelly_mult * bankroll


def edge(model_p: float, decimal_odds: float) -> float:
    """Your model's probability minus the book's implied probability."""
    return model_p - implied_prob(decimal_odds)


def format_bet(
    home_team: str,
    away_team: str,
    model_p: float,
    decimal_odds: float,
    bankroll: float,
    consensus: str,
    model_breakdown: dict[str, float],
) -> str:
    """Return a formatted string for one game's betting analysis."""
    imp   = implied_prob(decimal_odds)
    ev    = expected_value(model_p, decimal_odds)
    kf    = kelly_fraction(model_p, decimal_odds)
    stake_full = kf * bankroll
    stake_half = kf * 0.5 * bankroll
    edg   = edge(model_p, decimal_odds)
    pred  = "WIN" if model_p >= 0.5 else "LOSS"
    ev_flag = "✓ BET" if ev > 0 else "✗ SKIP"

    lines = [
        f"  {'─'*64}",
        f"  {home_team} (home) vs {away_team}",
        f"  {'─'*64}",
        f"  Ensemble probability : {model_p:>6.1%}  →  predicted {pred}",
        f"  Model consensus      : {consensus}",
    ]

    if model_breakdown:
        lines.append("  Per-window breakdown :")
        for label, p in sorted(model_breakdown.items()):
            lines.append(f"    {label:<30} {p:.1%}")

    lines += [
        "",
        f"  Decimal odds         : {decimal_odds:.2f}",
        f"  Book implied prob    : {imp:.1%}",
        f"  Your edge            : {edg:+.1%}",
        f"  Expected value       : {ev:+.1%}   {ev_flag}",
    ]

    if ev > 0:
        lines += [
            f"  Full Kelly stake     : {kf:.1%}   (${stake_full:,.2f})",
            f"  Half Kelly stake     : {kf*0.5:.1%}   (${stake_half:,.2f})  ← recommended",
        ]

    return "\n".join(lines)
