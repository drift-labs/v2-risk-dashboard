
# **Risk Dashboard Trial Assignment**

## **Objective**

Extend Drift Protocol's **Risk Dashboard** with a new column named **Target Scale IAW** (Initial Asset Weight). This column will calculate and display a value based on whether a **spot asset meets safety criteria**. If **all criteria pass**, the **Target Scale IAW** should be set to:

> **1.2x total deposits notional**

This assignment serves as an introduction to Drift Protocol's tech stack and its risk management tools in a **non-production environment**. The **Risk Dashboard** is primarily a back-end tool and involves interpreting, validating, and displaying risk-related data.

---

## **Criteria for Passing**

You will implement individual **Pass/Fail** columns for each of the following criteria to determine if the **Target Scale IAW** can be set. These columns should be clear, concise, and easily understandable. The criteria are:

### **1. On-Chain Liquidity Check**
- **Condition**: Simulate a swap of one of the largest user account positions using Jupiter, measuring the **price impact %**.
- **Pass Rule**: 
	- Pass if: price impact < (1 - [[20241223-160016 Maintenance Asset Weight|Maintenance Asset Weight]]).

---

### **2. Effective Leverage (Spot Positions)**
- **Pass Rule**: 
	- Pass if: [[20241223-160343 Aggregate Effective Leverage|Aggregate Effective Leverage]] < (0.5 * [[20241223-160016 Maintenance Asset Weight|Maintenance Asset Weight]]).

---

### **3. Effective Leverage (Perp Positions)**
- **Pass Rule**: 
	- Pass if: 1x <= [[20241223-161359 Effective Leverage|Effective Leverage]] <= 2x.

---

### **4. [[20241223-170104 Excess Leverage Coverage|Excess Leverage Coverage (Perp Market Insurance)]]**
- **Condition**: Filter for users whose leverage exceeds 2x.
- **Pass Rule**: 
	- Pass if: filtering for users with leverage > 2 have excess notional fully covered by the perp market's insurance fund.

---

## **Implementation Steps**

### **1. Add the Target Scale IAW Column**
- Create a new column in the **Risk Dashboard** called **Target Scale IAW**.
- Set the value to **1.2x total deposits notional** if all four safety criteria columns pass.
- If any of the criteria fail, display **N/A** or an appropriate fallback value, such as **"Criteria not met"**.

---

### **2. Add Individual Pass/Fail Columns**
- Implement four separate columns, one for each criterion listed above.
- Each column should display either **"Pass"** or **"Fail"** based on whether the corresponding rule is satisfied.
- Example column names:
  - **On-Chain Liquidity Pass**
  - **Spot Effective Leverage Pass**
  - **Perp Effective Leverage Pass**
  - **Excess Leverage Coverage Pass**

---

### **3. Expandable Tooltips for Detailed Explanations**
- Add a tooltip for each criterion that explains:
  - The condition being tested.
  - The mathematical formula or logic used.
  - Data sources and assumptions.
  - Examples to clarify edge cases or common failure scenarios.
- Example:
  - **On-Chain Liquidity Check Tooltip**:
    - "This column evaluates whether simulating the swap of a large user position results in a price impact that exceeds the acceptable threshold. The acceptable threshold is derived from (1 - maint asset weight). Data is sourced from Jupiter."

---

## **Explicit Engineering Considerations**

### **Data Source Integration**
- Ensure the integration with on-chain protocols like **Jupiter** for simulating swaps is robust and handles errors (e.g., API failures or unexpected data).
- Confirm that the **maint asset weight** is accurately retrieved from the appropriate Drift Protocol market specs.

### **Edge Cases**
- **Criteria failure**: Handle scenarios where one or more criteria fail. Ensure the **Target Scale IAW** column outputs a clear and user-friendly fallback value.
- **Data unavailability**: Implement fail-safes to handle missing or incomplete data, with error messaging/logging.

### **Testing Requirements**
- Write unit tests for each criterion to validate correct Pass/Fail outcomes based on sample data.
- Include test cases for:
  - Boundary values (e.g., effective leverage exactly at 1x or 2x).
  - Missing or malformed data inputs.

### **Performance**
- Ensure calculations and queries are efficient, as the Risk Dashboard may involve real-time updates for multiple assets and user accounts.

---

## **Example Outputs**

### **Before**
| User | Target Scale IAW | Comments |
|------|-------------------|----------|
| Bob  | N/A               | Missing criteria details |

### **After**
| User  | Target Scale IAW | On-Chain Liquidity Pass | Spot Leverage Pass | Perp Leverage Pass | Excess Coverage Pass |
| ----- | ---------------- | ----------------------- | ------------------ | ------------------ | -------------------- |
| Bob   | 1.2x             | Pass                    | Pass               | Pass               | Pass                 |
| Alice | N/A              | Fail                    | Pass               | Pass               | Fail                 |

---

This structure ensures clarity, supports debugging, and aligns with best practices for robust engineering deliverables. Let me know if further refinements or additional examples are needed.
