"""
Utility functions - formatting and helpers.
"""


def format_currency(value: float, symbol: str = "$") -> str:
    """
    Format a number as currency.

    Args:
        value: The numeric value
        symbol: Currency symbol (default: $)

    Returns:
        Formatted currency string (e.g., "$1,234.56")
    """
    if value is None:
        return "N/A"
    return f"{symbol}{value:,.2f}"


def format_large_number(value: float) -> str:
    """
    Format large numbers with K, M, B suffixes.

    Args:
        value: The numeric value

    Returns:
        Formatted string (e.g., "1.23M", "456K")
    """
    if value is None:
        return "N/A"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,.0f}"


def format_percent(value: float) -> str:
    """
    Format a number as percentage with sign.

    Args:
        value: The percentage value

    Returns:
        Formatted percentage string (e.g., "+2.34%", "-1.50%")
    """
    if value is None:
        return "N/A"

    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"
