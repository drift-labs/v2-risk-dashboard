---
aliases:
  - Excess Leverage Coverage (Perp Market Insurance)
  - Perp Market Insurance
  - Excess Leverage Coverage
---
___
# Body/Content

**Excess Leverage Coverage** evaluates whether a perpetual futures market has sufficient insurance fund reserves to cover the risk posed by highly leveraged traders. Specifically, it ensures that users whose leverage exceeds a critical threshold (e.g., **2x leverage**) are adequately backed by the insurance fund to prevent cascading liquidations or losses for the protocol.

---
## **Key Concepts**

1. **Perpetual Futures (Perps)**:
	 - A derivative product allowing traders to take leveraged positions on an asset without expiry.
	 - Traders can amplify gains and losses using leverage.
2. **Leverage**:
	 - Defined as:
		 - Leverage(x) = [[20241223-162925 Position Notional|Position Notional]]/[[20241223-164141 Collateral|Collateral]]
		 - Higher leverage means greater exposure relative to collateral, increasing both potential returns and risks.
3. **Insurance Fund**:
	 - A reserve maintained by the protocol to absorb losses from liquidations that fail to cover a trader’s debt.
	 - Acts as a safety net to protect the system from becoming under-collateralized.
4. **Excess Leverage**:
	 - Refers to users whose leverage exceeds a pre-defined threshold (e.g., **2x leverage**). These users pose a higher risk of liquidation, especially during volatile market conditions.
5. **Excess Notional**:
	 - The portion of a trader’s position that exceeds the leverage threshold, calculated as:
		 - Excess Notional = Position Notional - (Collateral x Leverage Threshold)

---
## **Purpose of Excess Leverage Coverage**

Excess Leverage Coverage ensures that:

1. The **insurance fund** has enough reserves to cover the potential shortfall from liquidating high-risk positions.
2. The protocol remains solvent, even during sharp market movements or unexpected events.

Without adequate coverage, a wave of liquidations could deplete the insurance fund and result in losses being passed on to other traders or the protocol itself.

___
## **How Excess Leverage Coverage Works**
1. **Identify High-Risk Users**:
	 - Filter users whose leverage exceeds the defined threshold (e.g., 2x leverage).
	 - These users are flagged as having “excess leverage.”
2. **Calculate Excess Notional**:
	 - For each flagged user, determine the excess notional portion of their position:
		 - Excess Notional = Position Notional - (Collateral x Leverage Threshold)
3. **Compare Against Insurance Fund**:
	 - Sum up the **excess notional** for all flagged users.
	 - Check if the **insurance fund** has enough reserves to cover this total.
4. **Pass/Fail Criteria**:
	 - **Pass**: If the insurance fund fully covers the total excess notional.
	 - **Fail**: If the total excess notional exceeds the insurance fund reserves.

---
## **Example Calculation**
**User Details**
- User’s position notional: **$100,000**
- User’s collateral: **$20,000**
- Leverage threshold: **2x**
- Insurance fund reserves: **$50,000**

**Step 1: Calculate User’s Leverage**

`Leverage = Position Notional / Collateral = 100,000/20,000 = 5x`

The user’s leverage exceeds the threshold of 2x, so they are flagged.

**Step 2: Calculate Excess Notional**

`Excess Notional = Position Notional - (Collateral x Leverage Threshold)`

`Excess Notional = 100,000 - (20,000 x 2) = (100,000 - 40,000) = 60,000`

**Step 3: Check Against Insurance Fund**

 - Total Excess Notional for all users: **$60,000**
 - Insurance Fund Reserves: **$50,000**

Result: The insurance fund cannot fully cover the excess notional, so this criterion **fails**.

---
## **Importance in Drift Protocol**

In Drift Protocol, **Excess Leverage Coverage** ensures:

1. **Systemic Stability**:
	 - By maintaining sufficient reserves in the insurance fund, the protocol minimizes the risk of cascading liquidations.
2. **Risk Management**:
	 - Helps identify high-risk users early and ensure their impact on the system is manageable.
3. **User Protection**:
	 - Provides confidence to traders that the platform is robust, even during extreme market events.

---
## **Engineering Considerations**

1. **Data Sources**:

• Retrieve user position notional, collateral, and leverage data in real-time.

• Access the current balance of the insurance fund.

2. **Performance**:

• Calculating excess leverage for all users can be resource-intensive. Optimize by focusing only on high-leverage accounts.

3. **Edge Cases**:

• **Zero Collateral**: Handle accounts with zero collateral to avoid division by zero.

• **Rapid Liquidations**: Account for changes in insurance fund reserves during high volatility.

4. **Visualization**:

• Add a column in the dashboard to indicate whether excess notional is covered.

• Include a tooltip explaining how excess leverage is calculated and the role of the insurance fund.

---
## **Example Code**

### **Filter High-Leverage Users**
```python
def get_excess_notional(users, leverage_threshold):
    """
    Calculate excess notional for users exceeding the leverage threshold.
    """
    excess_notional = []
    for user in users:
        notional = user['position_notional']
        collateral = user['collateral']
        leverage = notional / collateral if collateral > 0 else float('inf')

        if leverage > leverage_threshold:
            excess = notional - (collateral * leverage_threshold)
            excess_notional.append(max(0, excess))
    return sum(excess_notional)
```
### **Check Against Insurance Fund**
```python
def check_insurance_coverage(users, leverage_threshold, insurance_fund_reserves):
    """
    Determine if the insurance fund covers excess notional.
    """
    total_excess_notional = get_excess_notional(users, leverage_threshold)
    return total_excess_notional <= insurance_fund_reserves
```
### **Example Usage**
```python
users = [
    {"position_notional": 100000, "collateral": 20000},  # 5x leverage
    {"position_notional": 50000, "collateral": 25000},   # 2x leverage
]

insurance_fund_reserves = 60000
leverage_threshold = 2  # 2x

result = check_insurance_coverage(users, leverage_threshold, insurance_fund_reserves)
print("Insurance Coverage Pass" if result else "Insurance Coverage Fail")
# Output: Insurance Coverage Fail
```

---
## **Key Takeaways**

1. **Excess Leverage Coverage is critical for platform stability**:
	 - Ensures that highly leveraged traders don’t put the entire protocol at risk.
2. **Insurance funds act as the last line of defense**:
	 - A well-funded insurance pool prevents cascading failures during liquidation events.
3. **Proactive monitoring of high-leverage accounts is essential**:
	 - Filtering for excess leverage allows early intervention and risk mitigation.

---
Footer/References

___
Tags

___