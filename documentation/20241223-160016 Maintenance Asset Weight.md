---
aliases:
  - Maintenance Asset Weight
---
___
Body/Content

The **“maint asset weight”** (short for “maintenance asset weight”) is a parameter used in risk management within trading protocols like Drift to determine the margin or capital required to maintain a position. It is typically defined as a fraction (or percentage) of the position’s notional value and represents the minimum capital a user must have in their account to avoid liquidation.
  

**Context for Drift Protocol**

• **Maint asset weight** helps the protocol evaluate the risk associated with a position and ensures that the platform remains solvent during volatile market conditions.

• The weight is often specified in the protocol’s documentation for each market or asset class. For example:

• A maint asset weight of 0.8 means 80% of the position’s notional value is required as collateral to avoid liquidation.

• A maint asset weight of 1 means the full value of the position is required as collateral.

  

**In This Assignment**

  

The **“maint asset weight”** is part of the condition for **On-Chain Liquidity Check**:

  

  

\text{price impact } < (1 - \text{maint asset weight})

  

• **Interpretation**:

• The formula evaluates whether the price impact from a simulated swap is acceptable, compared to a threshold derived from the maint asset weight.

• A smaller maint asset weight indicates that the protocol tolerates less price impact, as more margin is required.

  

**Example**

• If the maint asset weight is **0.9**, then:

  

\text{price impact must be less than } (1 - 0.9) = 0.1 \, \text{(10%)}.

  

• This means the swap should result in no more than a 10% price impact for the criterion to pass.

  

**Where to Find Maint Asset Weight**

• The value for the maint asset weight is typically specified in the protocol’s market configuration or documentation. In the context of Drift Protocol, refer to the [Market Specs](https://docs.drift.trade/trading/market-specs) for details.

  

Let me know if you need further clarification!

___
Footer/References

___
Tags

___