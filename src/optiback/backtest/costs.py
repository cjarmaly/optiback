"""Transaction cost and slippage models for backtesting."""

from __future__ import annotations


def calculate_transaction_cost(
    trade_size: float,
    price: float,
    cost_per_share: float = 0.01,
    bid_ask_spread_bps: float = 5.0,
) -> float:
    """
    Calculate transaction cost including commissions and bid-ask spread.

    The cost is always positive regardless of trade direction (buy or sell).

    Args:
        trade_size: Number of shares traded (positive for buy, negative for sell)
        price: Execution price per share
        cost_per_share: Fixed cost per share (commissions, default: $0.01)
        bid_ask_spread_bps: Bid-ask spread in basis points (default: 5 bps = 0.05%)

    Returns:
        Total transaction cost in dollars (always positive)

    Examples:
        >>> cost = calculate_transaction_cost(trade_size=100, price=100.0, cost_per_share=0.01, bid_ask_spread_bps=5.0)
        >>> round(cost, 2)
        1.50
    """
    # Absolute trade value
    trade_value = abs(trade_size) * price

    # Fixed commission cost
    commission = abs(trade_size) * cost_per_share

    # Bid-ask spread cost (half spread on each side)
    spread_cost = trade_value * (bid_ask_spread_bps / 10000.0) / 2.0

    return commission + spread_cost


def apply_slippage(
    price: float,
    trade_size: float,
    slippage_bps: float = 5.0,
    impact_factor: float = 0.1,
) -> float:
    """
    Apply slippage to execution price based on trade size.

    Slippage represents the price movement against the trader:
    - Buying increases the price (you pay more)
    - Selling decreases the price (you receive less)

    Larger trades have more slippage due to market impact.

    Args:
        price: Reference price (mid-price)
        trade_size: Number of shares traded (positive for buy, negative for sell)
        slippage_bps: Base slippage in basis points (default: 5 bps = 0.05%)
        impact_factor: Market impact factor (default: 0.1, meaning 10% of base slippage per 100 shares)

    Returns:
        Execution price after slippage

    Examples:
        >>> exec_price = apply_slippage(price=100.0, trade_size=100, slippage_bps=5.0)
        >>> exec_price > 100.0  # Buying pushes price up
        True
    """
    # Calculate market impact based on trade size
    # Larger trades have more impact
    size_impact = 1.0 + (abs(trade_size) / 100.0) * impact_factor

    # Base slippage in decimal
    slippage_decimal = slippage_bps / 10000.0

    # Apply slippage: buying increases price, selling decreases price
    if trade_size > 0:
        # Buying: pay more (price goes up)
        return price * (1.0 + slippage_decimal * size_impact)
    elif trade_size < 0:
        # Selling: receive less (price goes down)
        return price * (1.0 - slippage_decimal * size_impact)
    else:
        # No trade
        return price


def calculate_total_execution_cost(
    trade_size: float,
    reference_price: float,
    cost_per_share: float = 0.01,
    bid_ask_spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
    impact_factor: float = 0.1,
) -> tuple[float, float]:
    """
    Calculate total execution cost including transaction costs and slippage.

    Args:
        trade_size: Number of shares traded (positive for buy, negative for sell)
        reference_price: Reference price (mid-price) before slippage
        cost_per_share: Fixed cost per share (default: $0.01)
        bid_ask_spread_bps: Bid-ask spread in basis points (default: 5 bps)
        slippage_bps: Base slippage in basis points (default: 5 bps)
        impact_factor: Market impact factor (default: 0.1)

    Returns:
        Tuple of (execution_price_with_slippage, total_transaction_cost)

    Examples:
        >>> exec_price, cost = calculate_total_execution_cost(
        ...     trade_size=100, reference_price=100.0
        ... )
        >>> exec_price > 100.0  # Slippage increases price for buy
        True
        >>> cost > 0  # Transaction cost is positive
        True
    """
    # Apply slippage to get execution price
    execution_price = apply_slippage(
        price=reference_price,
        trade_size=trade_size,
        slippage_bps=slippage_bps,
        impact_factor=impact_factor,
    )

    # Calculate transaction cost using execution price
    transaction_cost = calculate_transaction_cost(
        trade_size=trade_size,
        price=execution_price,
        cost_per_share=cost_per_share,
        bid_ask_spread_bps=bid_ask_spread_bps,
    )

    return execution_price, transaction_cost
