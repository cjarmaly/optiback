"""Tests for backtesting functionality."""

import numpy as np
import pytest

from optiback.backtest.costs import (
    apply_slippage,
    calculate_total_execution_cost,
    calculate_transaction_cost,
)
from optiback.backtest.delta_hedge import backtest_delta_hedge
from optiback.backtest.engine import BacktestResult, calculate_sharpe_ratio
from optiback.backtest.mispricing import backtest_mispricing


class TestTransactionCosts:
    """Test transaction cost calculations."""

    def test_transaction_cost_basic(self):
        """Test basic transaction cost calculation."""
        cost = calculate_transaction_cost(
            trade_size=100, price=100.0, cost_per_share=0.01, bid_ask_spread_bps=5.0
        )
        # Commission: 100 * 0.01 = 1.0
        # Spread: 100 * 100 * 0.0005 / 2 = 2.5
        # Total: 3.5
        assert abs(cost - 3.5) < 0.01

    def test_transaction_cost_sell(self):
        """Test transaction cost for sell trade."""
        cost_buy = calculate_transaction_cost(trade_size=100, price=100.0)
        cost_sell = calculate_transaction_cost(trade_size=-100, price=100.0)
        # Should be same (absolute value)
        assert abs(cost_buy - cost_sell) < 1e-10

    def test_transaction_cost_zero_trade(self):
        """Test transaction cost for zero trade."""
        cost = calculate_transaction_cost(trade_size=0, price=100.0)
        assert cost == 0.0

    def test_transaction_cost_custom_parameters(self):
        """Test transaction cost with custom parameters."""
        cost = calculate_transaction_cost(
            trade_size=50, price=200.0, cost_per_share=0.02, bid_ask_spread_bps=10.0
        )
        # Commission: 50 * 0.02 = 1.0
        # Spread: 50 * 200 * 0.001 / 2 = 5.0
        # Total: 6.0
        assert abs(cost - 6.0) < 0.01


class TestSlippage:
    """Test slippage calculations."""

    def test_slippage_buy(self):
        """Test slippage for buy trade increases price."""
        exec_price = apply_slippage(
            price=100.0, trade_size=100, slippage_bps=5.0, impact_factor=0.1
        )
        assert exec_price > 100.0  # Buying pushes price up

    def test_slippage_sell(self):
        """Test slippage for sell trade decreases price."""
        exec_price = apply_slippage(
            price=100.0, trade_size=-100, slippage_bps=5.0, impact_factor=0.1
        )
        assert exec_price < 100.0  # Selling pushes price down

    def test_slippage_zero_trade(self):
        """Test slippage for zero trade."""
        exec_price = apply_slippage(price=100.0, trade_size=0)
        assert exec_price == 100.0

    def test_slippage_larger_trade(self):
        """Test that larger trades have more slippage."""
        small_slippage = apply_slippage(price=100.0, trade_size=100) - 100.0
        large_slippage = apply_slippage(price=100.0, trade_size=500) - 100.0
        assert large_slippage > small_slippage


class TestTotalExecutionCost:
    """Test total execution cost calculation."""

    def test_total_execution_cost_buy(self):
        """Test total execution cost for buy."""
        exec_price, cost = calculate_total_execution_cost(
            trade_size=100, reference_price=100.0, slippage_bps=5.0
        )
        assert exec_price > 100.0  # Slippage increases price
        assert cost > 0  # Transaction cost is positive

    def test_total_execution_cost_sell(self):
        """Test total execution cost for sell."""
        exec_price, cost = calculate_total_execution_cost(
            trade_size=-100, reference_price=100.0, slippage_bps=5.0
        )
        assert exec_price < 100.0  # Slippage decreases price
        assert cost > 0  # Transaction cost is positive

    def test_total_execution_cost_zero(self):
        """Test total execution cost for zero trade."""
        exec_price, cost = calculate_total_execution_cost(trade_size=0, reference_price=100.0)
        assert exec_price == 100.0
        assert cost == 0.0


class TestDeltaHedgeBacktest:
    """Test delta-hedge backtesting."""

    def test_delta_hedge_basic_call(self):
        """Test basic delta-hedge backtest for call option."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        result = backtest_delta_hedge(
            spot_prices=spot_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            option_position=-1.0,  # Short call
        )

        assert isinstance(result, BacktestResult)
        assert result.strategy_type == "delta_hedge"
        assert result.num_trades >= 0
        assert result.transaction_costs >= 0
        assert result.slippage_costs >= 0

    def test_delta_hedge_basic_put(self):
        """Test basic delta-hedge backtest for put option."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        result = backtest_delta_hedge(
            spot_prices=spot_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="put",
            option_position=-1.0,  # Short put
        )

        assert isinstance(result, BacktestResult)
        assert result.strategy_type == "delta_hedge"
        assert result.num_trades >= 0

    def test_delta_hedge_single_price(self):
        """Test delta-hedge with single price raises error."""
        spot_prices = np.array([100.0])
        with pytest.raises(ValueError, match="at least 2"):
            backtest_delta_hedge(
                spot_prices=spot_prices,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.25,
                option_type="call",
            )

    def test_delta_hedge_with_costs(self):
        """Test delta-hedge with custom cost parameters."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        result = backtest_delta_hedge(
            spot_prices=spot_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            cost_per_share=0.02,
            bid_ask_spread_bps=10.0,
            slippage_bps=10.0,
        )

        assert result.transaction_costs >= 0
        assert result.slippage_costs >= 0

    def test_delta_hedge_with_dividend(self):
        """Test delta-hedge with dividend yield."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        result = backtest_delta_hedge(
            spot_prices=spot_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            dividend_yield=0.03,
        )

        assert isinstance(result, BacktestResult)


class TestMispricingBacktest:
    """Test mispricing backtesting."""

    def test_mispricing_basic_black_scholes(self):
        """Test basic mispricing backtest with Black-Scholes."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        # Market prices slightly undervalued
        market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            theoretical_model="black_scholes",
            mispricing_threshold=0.05,
        )

        assert isinstance(result, BacktestResult)
        assert result.strategy_type == "mispricing"
        assert result.num_trades >= 0

    def test_mispricing_basic_put(self):
        """Test basic mispricing backtest for put option."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        market_prices = np.array([6.0, 5.5, 6.5, 5.0, 6.0])

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="put",
            theoretical_model="black_scholes",
        )

        assert isinstance(result, BacktestResult)

    def test_mispricing_binomial_model(self):
        """Test mispricing backtest with binomial model."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            theoretical_model="binomial",
        )

        assert isinstance(result, BacktestResult)

    def test_mispricing_monte_carlo_model(self):
        """Test mispricing backtest with Monte Carlo model."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            theoretical_model="monte_carlo",
            simulations=10000,
            seed=42,
        )

        assert isinstance(result, BacktestResult)

    def test_mispricing_mismatched_lengths(self):
        """Test mispricing backtest with mismatched price arrays."""
        spot_prices = np.array([100.0, 101.0, 99.0])
        market_prices = np.array([7.0, 7.5])

        with pytest.raises(ValueError, match="same length"):
            backtest_mispricing(
                spot_prices=spot_prices,
                market_option_prices=market_prices,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.25,
                option_type="call",
            )

    def test_mispricing_invalid_model(self):
        """Test mispricing backtest with invalid model."""
        spot_prices = np.array([100.0, 101.0])
        market_prices = np.array([7.0, 7.5])

        with pytest.raises(ValueError, match="black_scholes.*binomial.*monte_carlo"):
            backtest_mispricing(
                spot_prices=spot_prices,
                market_option_prices=market_prices,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.25,
                option_type="call",
                theoretical_model="invalid_model",
            )

    def test_mispricing_high_threshold(self):
        """Test mispricing backtest with high threshold (fewer trades)."""
        spot_prices = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.25,
            option_type="call",
            mispricing_threshold=0.50,  # Very high threshold
        )

        assert result.num_trades >= 0


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        """Test creating a BacktestResult."""
        result = BacktestResult(
            total_pnl=100.0,
            transaction_costs=10.0,
            slippage_costs=5.0,
            num_trades=5,
            initial_value=1000.0,
            final_value=1100.0,
            returns=0.0,  # Will be calculated
            strategy_type="delta_hedge",
        )

        assert result.total_pnl == 100.0
        assert result.transaction_costs == 10.0
        assert result.returns == 0.1  # (1100 - 1000) / 1000

    def test_backtest_result_summary(self):
        """Test BacktestResult summary method."""
        result = BacktestResult(
            total_pnl=100.0,
            transaction_costs=10.0,
            slippage_costs=5.0,
            num_trades=5,
            initial_value=1000.0,
            final_value=1100.0,
            returns=0.0,
            strategy_type="delta_hedge",
        )

        summary = result.summary()
        assert summary["strategy"] == "delta_hedge"
        assert summary["total_pnl"] == 100.0
        assert summary["net_pnl"] == 85.0  # 100 - 10 - 5
        assert summary["returns"] == 0.1

    def test_backtest_result_zero_initial_value(self):
        """Test BacktestResult with zero initial value."""
        result = BacktestResult(
            total_pnl=100.0,
            transaction_costs=10.0,
            slippage_costs=5.0,
            num_trades=5,
            initial_value=0.0,
            final_value=100.0,
            returns=0.0,
            strategy_type="mispricing",
        )

        assert result.returns == 0.0  # Avoid division by zero


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_sharpe_ratio_zero_returns(self):
        """Test Sharpe ratio with zero returns."""
        returns = np.array([0.0, 0.0, 0.0])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_empty_returns(self):
        """Test Sharpe ratio with empty array."""
        returns = np.array([])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_constant_returns(self):
        """Test Sharpe ratio with constant returns."""
        returns = np.array([0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        # Should handle zero std dev gracefully
        assert isinstance(sharpe, float)
