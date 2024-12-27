---
aliases:
  - Aggregate Effective Leverage
---
___
Body/Content

**Aggregate Effective Leverage** is a metric used in trading systems like Drift Protocol to measure the risk exposure of an account relative to its collateral. It accounts for the combined effect of all positions (spot and derivative) in an account, scaled by their notional value, and evaluates the leverage at an aggregate level.

**Key Components of Aggregate Effective Leverage**

1. **[[20241223-161359 Effective Leverage|Effective Leverage]]:**
	• Defined as:
		[[20241223-161359 Effective Leverage|Effective Leverage]] = ([[20241223-162925 Position Notional|Position Notional]]/[[20241223-164141 Collateral|Collateral]])
		It represents the ratio of the notional value of a position (or group of positions) to the collateral backing it.

2. **Aggregate Notional Value:**
	• The combined notional value of all positions in the account. Notional value is the dollar value of the assets represented by the positions.

3. **Aggregate Collateral:**
	• The total amount of collateral available in the account to support all positions.

---

**Formula for Aggregate Effective Leverage**

Aggregate effective leverage sums up the effective leverage of individual positions, weighted by their respective notional values:

Aggregate Effective Leverage = (Sum of All Position Notional Values/Total Collateral)

Where:

• **Position Notional Values**: The dollar equivalent of each position (spot and derivatives).

• **Total Collateral**: The total funds or assets in the account used to back all positions.


---

**In This Assignment**

The **Effective Leverage (Spot Positions)** criterion specifies that the **Aggregate Effective Leverage** for spot positions must satisfy:

Aggregate Effective Leverage < (0.5 x [[20241223-160016 Maintenance Asset Weight|Maintenance Asset Weight]])

This rule ensures that the user’s overall leverage remains below a risk threshold relative to the asset’s maintenance weight. It prevents over-leveraging in spot markets.

---
**Example Calculation**
1. **User Account Details:**
	• Total collateral: **$10,000**
		• Spot Position 1: Notional value **$5,000**
		• Spot Position 2: Notional value **$3,000**
	
2. **Aggregate Effective Leverage:**
	Aggregate Effective Leverage = (Total Notional Value of Spot Positions/Total Collateral)
	Aggregate Effective Leverage = ((5000 + 3000)/(10000)) = 0.8
3. **Pass Criteria:**
	• If the **maint asset weight** is **0.9**, the pass condition becomes:
		Aggregate Effective Leverage < (0.5 x 0.9) = 0.45
	• In this case, **0.8 > 0.45**, so the criterion **fails**.

---

**Engineering Considerations**

1. **Data Sources:**
	• Ensure accurate retrieval of **notional values** for all spot positions and total collateral from the user’s account.
2. **Edge Cases:**
	• **No positions**: If the user has no active spot positions, the effective leverage should default to **0** (passing the criterion).
	• **Insufficient collateral**: Handle cases where collateral is missing or zero to avoid division by zero errors.
3. **Testing:**
	• Validate with accounts having:
		• No positions.
		• High leverage with one or multiple positions.
		• Boundary cases where leverage is exactly at the threshold.

___
Footer/References

___
Tags

___