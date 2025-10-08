import pytest
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calculations import *


"""Test cases for calculate_daily_returns function"""

# Happy Path Tests - Normal market data
def test_daily_returns_happy_path_positive_numbers():
    """Test with normal positive price data"""
    close_prices = [100.50, 105.25, 102.75, 108.80]
    expected = [0, 4.72636815920398, -2.375296912114014, 5.888077858880779]
    result = calculate_daily_returns(close_prices)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Boundary Tests - Single element list
def test_daily_returns_single_element():
    """Test with only one price point"""
    close_prices = [50.75]
    expected = [0]
    result = calculate_daily_returns(close_prices)
    assert result == expected

# Boundary Tests - Two elements
def test_daily_returns_two_elements():
    """Test with exactly two price points"""
    close_prices = [100.25, 110.50]
    expected = [0, 10.224438902743142]
    result = calculate_daily_returns(close_prices)
    assert math.isclose(result[1], expected[1], rel_tol=1e-9)

# Edge Case Tests - Empty list (Invalid)
def test_daily_returns_empty_list():
    """Test with empty price list"""
    close_prices = []
    with pytest.raises(ValueError, match="Close prices list cannot be empty"):
        calculate_daily_returns(close_prices)

# Edge Case Tests - Zero price (handle division by zero) (Invalid)
def test_daily_returns_zero_price_division_by_zero():
    """Test with zero price which causes division by zero"""
    close_prices = [100.50, 0.00, 50.25]
    
    # This should raise a division by zero error
    with pytest.raises(ZeroDivisionError, match="Stock prices cannot be zero"):
        calculate_daily_returns(close_prices)

# Edge Case Tests - Negative prices (Invalid)
def test_daily_returns_negative_prices():
    """Test with negative prices which are invalid for stock prices"""
    close_prices = [100.50, -50.25, 75.75]
    
    with pytest.raises(ValueError, match="Stock prices cannot be negative"):
        calculate_daily_returns(close_prices)

# Edge Case Tests - Very small numbers
def test_daily_returns_very_small_numbers():
    """Test with very small price values"""
    close_prices = [0.0000005, 0.00000075, 0.000000025]
    expected = [0, 50.000000000000014, -96.66666666666667]
    result = calculate_daily_returns(close_prices)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Edge Case Tests - Very large numbers
def test_daily_returns_very_large_numbers():
    """Test with very large price values"""
    close_prices = [1000000.50, 1100000.75, 900000.25]
    expected = [0, 10.00002, -18.181851239646882]
    result = calculate_daily_returns(close_prices)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Edge Case Tests - Identical prices (zero returns)
def test_daily_returns_identical_prices():
    """Test with identical consecutive prices"""
    close_prices = [50.25, 50.25, 50.25, 50.25]
    expected = [0, 0.0, 0.0, 0.0]
    result = calculate_daily_returns(close_prices)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)



"""Test cases for calculate_true_range function"""

# Happy Path Tests - Normal market data
def test_true_range_happy_path_normal_data():
    """Test with normal high, low, close data"""
    high = [105.25, 108.75, 104.50, 110.80]
    low = [95.75, 102.25, 100.50, 105.25]
    close = [100.50, 105.25, 102.75, 108.80]
    expected = [9.50, 8.25, 4.75, 8.05]
    result = calculate_true_range(high, low, close)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Boundary Tests - Single day
def test_true_range_single_day():
    """Test with single day of data"""
    high = [100.75]
    low = [90.25]
    close = [95.50]
    expected = [10.50]
    result = calculate_true_range(high, low, close)
    assert result == expected

# Edge Case Tests - Empty lists (Invalid)
def test_true_range_empty_lists():
    """Test with empty price lists"""
    with pytest.raises(ValueError, match="All price lists.*must be non-empty"):
        calculate_true_range([], [], [])
    
    with pytest.raises(ValueError, match="All price lists.*must be non-empty"):
        calculate_true_range([100.50], [], [95.50])

# Edge Case Tests - Zero price (Invalid)
def test_true_range_zero_price():
    """Test with zero price"""
    high = [100.50, 55.25, 75.75]
    low = [90.25, 0, 70.50]
    close = [95.75, 50.25, 72.25]
    
    with pytest.raises(ValueError, match="Stock prices cannot be zero"):
        calculate_true_range(high, low, close)

# Edge Case Tests - Negative prices (Invalid)
def test_true_range_negative_prices_tr():
    """Test with negative prices which should raise error"""
    high = [100.50, -50.25, 75.75]
    low = [90.25, -60.75, 70.50]
    close = [95.75, -55.25, 72.25]
    
    with pytest.raises(ValueError, match="Stock prices cannot be negative"):
        calculate_true_range(high, low, close)

# Edge Case Tests - Different length lists (Invalid)
def test_true_range_different_length_lists():
    """Test with lists of different lengths"""
    high = [105.25, 108.75]
    low = [95.75]
    close = [100.50, 105.25, 102.75]
    
    with pytest.raises(ValueError, match="All price lists must have the same length"):
        calculate_true_range(high, low, close)

# Edge Case Tests - High price less than low price (Invalid)
def test_true_range_high_less_than_low():
    """Test when high price is less than low price"""
    high = [100.50, 95.25]  # Day 2: high < low
    low = [90.25, 100.50]
    close = [95.75, 98.25]
    
    with pytest.raises(ValueError, match="High price.*cannot be less than low price"):
        calculate_true_range(high, low, close)

# Edge Case Tests - Low price more than close price (Invalid)
def test_true_range_low_more_than_close():
    """Test when low price is more than close price"""
    high = [100.50, 100.50]  # Day 2: low > close
    low = [90.25, 98.25]
    close = [95.75, 95.25]
    
    with pytest.raises(ValueError, match="Low price.*cannot be more than close price"):
        calculate_true_range(high, low, close)

# Edge Case Tests - Close price more than high price (Invalid)
def test_true_range_close_more_than_high():
    """Test when close price is more than high price"""
    high = [100.50, 98.25]  # Day 2: close > high
    low = [90.25, 95.25]
    close = [95.75, 100.50]
    
    with pytest.raises(ValueError, match="Close price.*cannot be more than high price"):
        calculate_true_range(high, low, close)

# Edge Case Tests - Zero range day
def test_true_range_zero_range():
    """Test when high equals low (zero range)"""
    high = [100.50, 105.25, 105.25]
    low = [90.75, 105.25, 105.25]
    close = [95.75, 105.25, 105.25]
    expected = [9.75, 9.5, 0.0]
    result = calculate_true_range(high, low, close)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)



"""Test cases for calculate_average_true_range function"""

# Happy Path Tests - Normal TR data
def test_average_true_range_happy_path_normal_tr():
    """Test with normal true range values"""
    tr_values = [10.50, 8.25, 12.75, 9.33, 11.67, 13.42, 7.89, 10.15, 14.28, 8.91]
    expected = [10.50, 9.375, 10.50, 10.2075, 10.50, 10.986666666666666, 10.544285714285715, 10.487959183673471, 11.029679300291546, 10.726867971678468]
    result = calculate_average_true_range(tr_values)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Boundary Tests - Single TR value
def test_average_true_range_single_tr_value():
    """Test with only one true range value"""
    tr_values = [15.75]
    expected = [15.75]
    result = calculate_average_true_range(tr_values)
    assert result == expected

# Boundary Tests - Exactly 7 days
def test_average_true_range_exactly_seven_days():
    """Test with exactly 7 days of data"""
    tr_values = [10.25, 12.75, 8.50, 14.25, 9.75, 11.25, 13.50]
    expected = [10.25, 11.50, 10.50, 11.4375, 11.10, 11.125, 11.464285714285714]
    result = calculate_average_true_range(tr_values)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Edge Case Tests - Empty TR list (Invalid)
def test_average_true_range_empty_tr_list():
    """Test with empty true range list"""
    tr_values = []
    with pytest.raises(ValueError, match="True range list cannot be empty"):
        calculate_average_true_range(tr_values)

# Edge Case Tests - All zero TR values
def test_average_true_range_all_zero_tr():
    """Test when all true range values are zero"""
    tr_values = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    expected = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    result = calculate_average_true_range(tr_values)
    assert result == expected

# Edge Case Tests - Negative TR values (Invalid)
def test_average_true_range_negative_tr_values():
    """Test with negative true range values (invalid data)"""
    tr_values = [10.50, -5.25, 8.75, -2.50, 12.25]
    
    with pytest.raises(ValueError, match="True range values cannot be negative"):
        calculate_average_true_range(tr_values)

# Edge Case Tests - Very small TR values
def test_average_true_range_very_small_tr():
    """Test with very small true range values"""
    tr_values = [0.0000001, 0.0000002, 0.00000015, 0.0000005]
    expected = [0.0000001, 0.00000015, 0.00000015, 0.0000002375]
    result = calculate_average_true_range(tr_values)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)

# Edge Case Tests - Very large TR values
def test_average_true_range_very_large_tr():
    """Test with very large true range values"""
    tr_values = [1000000.50, 2000000.75, 1500000.25, 500000.80]
    expected = [1000000.50, 1500000.625, 1500000.50, 1250000.575]
    result = calculate_average_true_range(tr_values)
    
    for i in range(len(expected)):
        assert math.isclose(result[i], expected[i], rel_tol=1e-9)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])