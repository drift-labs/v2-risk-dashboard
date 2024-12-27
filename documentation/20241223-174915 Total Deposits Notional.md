---
aliases:
  - Total Deposits Notional
  - Deposits Notional
---
___
# Body/Content
**Total Deposits Notional** represents the total dollar-equivalent value of all collateral deposited by a trader on a trading platform. It includes the combined value of all assets held by the trader, converted to a common denomination (usually USD or another stable value reference), and serves as the foundation for calculating margin, leverage, and risk metrics.

---
## **Key Characteristics of Total Deposits Notional**

1. **Aggregate Collateral Value**:
	 - It sums up the dollar-equivalent value of all deposited assets, including cryptocurrencies, stablecoins, or other eligible assets.
2. **Dynamic**:
	 - The value is dynamic and fluctuates with market prices for volatile assets. For instance, if a trader deposits BTC as collateral, the total deposits notional changes as the price of BTC changes.
3. **Common Denomination**:
	 - All deposited assets are converted to a common unit (e.g., USD) using their current market prices, enabling consistent calculations.
4. **Risk Management Input**:
	 - Used in risk calculations such as margin requirements, liquidation thresholds, and leverage limits.
5. **Scope of Usage**:
	 - In trading platforms like Drift Protocol, total deposits notional underpins calculations for metrics such as **Target Scale IAW**, **effective leverage**, and **margin utilization**.

---
## **Formula for Total Deposits Notional**
Total Deposits Notional} = \sum_{i=1}^{n} (\text{Asset Quantity}_i \times \text{Price}_i)
![[Pasted image 20241223175443.png]]
Where:
 - Asset Quantity_i: The quantity of the i^{th} deposited asset.
 - Price_i: The current market price of the i^{th} asset.
 - n: Total number of distinct assets deposited.

---
## **Examples of Total Deposits Notional**
### **Single Asset Example**
 - A trader deposits **2 BTC** as collateral, and the current BTC price is **$30,000**.

Total Deposits Notional = 2 x 30,000 = 60,000 USD

### **Multiple Asset Example**
 - A trader deposits the following assets:
	 - **2 BTC** at a price of **$30,000**
	 - **5 ETH** at a price of **$2,000**
	 - **10,000 USDC** (stablecoin, assumed to be pegged to $1).

Total Deposits Notional = (2 x 30,000) + (5 x 2,000) + (10,000 x 1)
Total Deposits Notional = (60,000 + 10,000 + 10,000) = 80,000 USD

---
## **How Total Deposits Notional is Used**
1. **Margin Requirements**:
	- Determines the [[20241223-164141 Collateral|Collateral]] available for opening and maintaining positions.
2. **Leverage Calculations**:
	- Serves as the denominator in effective leverage calculations:
		- [[20241223-161359 Effective Leverage|Effective Leverage]] = [[20241223-162925 Position Notional|Position Notional]] / [[20241223-174915 Total Deposits Notional|Total Deposits Notional]]
3. **Risk Metrics**:
	- Used to compute metrics like **[[20241223-173355 Target Scale IAW|Target Scale IAW]]:
		- [[20241223-173355 Target Scale IAW|Target Scale IAW]] = 1.2 x [[20241223-174915 Total Deposits Notional|Total Deposits Notional]] (if all criteria pass)}.
4. **Liquidation Thresholds**:
	- Helps set thresholds where positions are liquidated if the [[20241223-164141 Collateral|Collateral]] value drops below maintenance requirements.

---
## **Factors Impacting Total Deposits Notional**
1. **Market Volatility**:
	 - The notional value of volatile assets (e.g., BTC, ETH) changes with price fluctuations, directly affecting the **total deposits notional**.
2. **Deposits and Withdrawals**:
	 - Adding or removing assets from the account impacts the total deposits notional.
3. **Asset Eligibility**:
	 - Only eligible assets (as defined by the platform) are included in the calculation. For instance, assets with low liquidity or high volatility may not qualify.
4. **Exchange Rates**:
	 - Conversion between assets is dependent on the current exchange rate or price provided by the platform’s price oracle.

---
## **Engineering Considerations**
1. **Real-Time Updates**:
	 - Total deposits notional should be recalculated in real-time or near-real-time to account for market price changes, deposits, and withdrawals.
2. **Price Oracle Integration**:
	 - Ensure accurate and reliable price feeds for asset valuation. Handle scenarios like delayed or incorrect oracle data gracefully.
3. **Fallback Handling**:
	 - Provide a default behavior if an asset’s price is unavailable or if an oracle error occurs (e.g., exclude the asset from calculations or use the last known price).
4. **Edge Cases**:
	 - **Zero Deposits**: Handle accounts with no deposits, ensuring calculations default appropriately (e.g., total deposits notional is zero).
	 - **Highly Volatile Assets**: Apply safeguards to prevent drastic swings in total deposits notional due to flash crashes or temporary oracle anomalies.

---
## **Example Implementation**
### **Calculate Total Deposits Notional**
```python
def calculate_total_deposits_notional(assets: dict, price_oracle: dict) -> float:
    """
    Calculate the total deposits notional value in USD.

    Args:
        assets (dict): A dictionary of asset quantities. Example: {"BTC": 2, "ETH": 5, "USDC": 10000}.
        price_oracle (dict): A dictionary of asset prices. Example: {"BTC": 30000, "ETH": 2000, "USDC": 1}.

    Returns:
        float: The total deposits notional in USD.
    """
    total_notional = 0
    for asset, quantity in assets.items():
        price = price_oracle.get(asset, 0)  # Default price is 0 if not found
        total_notional += quantity * price
    return total_notional

# Example usage
assets = {"BTC": 2, "ETH": 5, "USDC": 10000}
price_oracle = {"BTC": 30000, "ETH": 2000, "USDC": 1}
total_notional = calculate_total_deposits_notional(assets, price_oracle)
print(f"Total Deposits Notional: ${total_notional}")
# Output: Total Deposits Notional: $80,000
```

---
## **Potential Questions and Edge Cases**
1. **What happens if an asset’s price is unavailable?**
	 - Exclude the asset from the calculation or use the last known price. Alternatively, mark the calculation as incomplete and notify the user.
2. **How does volatility affect total deposits notional?**
	 - Assets like BTC and ETH introduce significant variability. Stablecoins can be used to stabilize the total deposits notional.
3. **What about multiple collateral types?**
	 - Ensure all eligible assets are included, applying conversion rates for a consistent valuation.
4. **Can total deposits notional be negative?**
	 - No, as only deposited assets are counted. Withdrawals or fees reduce the notional but cannot result in a negative total.

---
## **Key Takeaways**
1. **Total Deposits Notional** reflects the combined dollar-equivalent value of all collateral deposited by a user.
2. It is a dynamic, real-time metric that changes with price fluctuations, deposits, and withdrawals.
3. It underpins critical risk and leverage calculations, ensuring the protocol maintains stability while maximizing collateral efficiency.

___
Footer/References

___
Tags

___