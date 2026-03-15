"""
Canadian tax calculation for trading profits.

Implements 2025 federal and provincial tax brackets.  Provides a single
function ``compute_trade_tax()`` used by the backtest engine to deduct
tax from each profitable trade.

Key rules:
- **Capital gains**: 50 % of the gain is included in taxable income.
  Effective rate = marginal_rate × 0.50.
- **Business income**: 100 % of the gain is taxable.
  Effective rate = marginal_rate × 1.00.
- **Losses**: Not taxed (tax = 0).  Capital losses carry forward to
  offset future capital gains; business losses are deductible.  We
  don't model loss-carry-forward inside a single backtest — we simply
  skip the tax on losing trades.

Auto-detect heuristic (when ``tax_treatment == "auto"``):
  If avg_bars_held < 5  OR  annualized_trades > 150 → business income.
  Otherwise → capital gains.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# 2025 Canadian federal tax brackets
# Source: https://www.canada.ca/en/revenue-agency/services/tax/individuals/
#         frequently-asked-questions-individuals/canadian-income-tax-rates-
#         individuals-current-previous-years.html
# ---------------------------------------------------------------------------

FEDERAL_BRACKETS: list[tuple[float, float]] = [
    # (upper_limit, rate)  — upper_limit = float("inf") for the top bracket
    (57_375.0, 0.15),
    (114_750.0, 0.205),
    (158_468.0, 0.26),
    (220_000.0, 0.29),
    (float("inf"), 0.33),
]

# ---------------------------------------------------------------------------
# 2025 Provincial tax brackets
# Each province maps to a list of (upper_limit, rate) tuples.
# ---------------------------------------------------------------------------

PROVINCIAL_BRACKETS: dict[str, list[tuple[float, float]]] = {
    "AB": [  # Alberta
        (148_269.0, 0.10),
        (177_922.0, 0.12),
        (237_230.0, 0.13),
        (355_845.0, 0.14),
        (float("inf"), 0.15),
    ],
    "BC": [  # British Columbia
        (47_937.0, 0.0506),
        (95_875.0, 0.077),
        (110_076.0, 0.105),
        (133_664.0, 0.1229),
        (181_232.0, 0.147),
        (252_752.0, 0.168),
        (float("inf"), 0.205),
    ],
    "MB": [  # Manitoba
        (47_000.0, 0.108),
        (100_000.0, 0.1275),
        (float("inf"), 0.174),
    ],
    "NB": [  # New Brunswick
        (49_958.0, 0.094),
        (99_916.0, 0.14),
        (185_064.0, 0.16),
        (float("inf"), 0.195),
    ],
    "NL": [  # Newfoundland and Labrador
        (43_198.0, 0.087),
        (86_395.0, 0.145),
        (154_244.0, 0.158),
        (215_943.0, 0.178),
        (275_870.0, 0.198),
        (551_739.0, 0.208),
        (1_103_478.0, 0.213),
        (float("inf"), 0.218),
    ],
    "NS": [  # Nova Scotia
        (29_590.0, 0.0879),
        (59_180.0, 0.1495),
        (93_000.0, 0.1667),
        (150_000.0, 0.175),
        (float("inf"), 0.21),
    ],
    "NT": [  # Northwest Territories
        (50_597.0, 0.059),
        (101_198.0, 0.086),
        (164_525.0, 0.122),
        (float("inf"), 0.1405),
    ],
    "NU": [  # Nunavut
        (53_268.0, 0.04),
        (106_537.0, 0.07),
        (173_205.0, 0.09),
        (float("inf"), 0.115),
    ],
    "ON": [  # Ontario
        (51_446.0, 0.0505),
        (102_894.0, 0.0915),
        (150_000.0, 0.1116),
        (220_000.0, 0.1216),
        (float("inf"), 0.1316),
    ],
    "PE": [  # Prince Edward Island
        (32_656.0, 0.098),
        (64_313.0, 0.138),
        (105_000.0, 0.167),
        (140_000.0, 0.175),
        (float("inf"), 0.19),
    ],
    "QC": [  # Quebec
        (51_780.0, 0.14),
        (103_545.0, 0.19),
        (126_000.0, 0.24),
        (float("inf"), 0.2575),
    ],
    "SK": [  # Saskatchewan
        (52_057.0, 0.105),
        (148_734.0, 0.125),
        (float("inf"), 0.145),
    ],
    "YT": [  # Yukon
        (55_867.0, 0.064),
        (111_733.0, 0.09),
        (154_906.0, 0.109),
        (500_000.0, 0.128),
        (float("inf"), 0.15),
    ],
}

# All valid province codes
VALID_PROVINCES = sorted(PROVINCIAL_BRACKETS.keys())

# Province display names
PROVINCE_NAMES: dict[str, str] = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland & Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}


# ---------------------------------------------------------------------------
# Tax computation helpers
# ---------------------------------------------------------------------------

def _marginal_rate(income: float, brackets: list[tuple[float, float]]) -> float:
    """Return the marginal tax rate for the given income.

    The *marginal* rate is the rate on the **next dollar** of income,
    which is the correct rate for taxing incremental trade profits.
    """
    for upper, rate in brackets:
        if income <= upper:
            return rate
    # Should not reach here since last bracket is inf, but just in case
    return brackets[-1][1]


def _effective_tax_on_amount(amount: float, brackets: list[tuple[float, float]]) -> float:
    """Compute the total tax owed on *amount* using progressive brackets.

    This is used when you want to compute tax from scratch (not used in
    the per-trade calculation, which uses marginal rate).
    """
    tax = 0.0
    prev_limit = 0.0
    for upper, rate in brackets:
        if amount <= prev_limit:
            break
        taxable = min(amount, upper) - prev_limit
        if taxable > 0:
            tax += taxable * rate
        prev_limit = upper
    return tax


def get_combined_marginal_rate(
    annual_income: float,
    province: str,
) -> float:
    """Return the combined federal + provincial marginal tax rate.

    Parameters
    ----------
    annual_income : float
        The taxpayer's annual income *before* trading profits.
    province : str
        Two-letter province code (e.g. "ON", "BC", "AB").

    Returns
    -------
    float
        Combined marginal rate (e.g. 0.4341 for ~43.41%).
    """
    fed_rate = _marginal_rate(annual_income, FEDERAL_BRACKETS)
    prov_brackets = PROVINCIAL_BRACKETS.get(province, PROVINCIAL_BRACKETS["ON"])
    prov_rate = _marginal_rate(annual_income, prov_brackets)
    return fed_rate + prov_rate


TaxTreatment = Literal["auto", "capital_gains", "business_income"]


def detect_tax_treatment(
    avg_bars_held: float,
    total_trades: int,
    trading_days: int,
) -> Literal["capital_gains", "business_income"]:
    """Auto-detect whether trading activity is capital gains or business income.

    Heuristic (based on CRA guidelines):
    - avg_bars_held < 5 trading days → business income (day/swing trading)
    - annualized_trades > 150 → business income (high frequency)
    - Otherwise → capital gains

    Parameters
    ----------
    avg_bars_held : float
        Average number of bars (trading days) each position was held.
    total_trades : int
        Total number of completed trades in the backtest.
    trading_days : int
        Total number of trading days in the backtest period.
    """
    if avg_bars_held < 5:
        return "business_income"

    if trading_days > 0:
        annualized_trades = total_trades * (252 / trading_days)
        if annualized_trades > 150:
            return "business_income"

    return "capital_gains"


def compute_trade_tax(
    pnl: float,
    marginal_rate: float,
    treatment: Literal["capital_gains", "business_income"],
) -> float:
    """Compute the tax owed on a single trade's profit.

    Parameters
    ----------
    pnl : float
        The trade's profit (positive) or loss (negative), after
        commissions and slippage.
    marginal_rate : float
        Combined federal + provincial marginal rate.
    treatment : str
        "capital_gains" (50 % inclusion) or "business_income" (100 %).

    Returns
    -------
    float
        Tax amount (>= 0).  Zero for losing trades.
    """
    if pnl <= 0:
        return 0.0

    if treatment == "capital_gains":
        inclusion_rate = 0.50
    else:
        inclusion_rate = 1.00

    return pnl * inclusion_rate * marginal_rate
