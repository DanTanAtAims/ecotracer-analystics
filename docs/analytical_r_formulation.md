# Analytical Formulation of the Residual Vector $\mathbf{r}$

In the Ecotracer reverse workflow, the system of equations for the parameter set $\theta$ is formulated as:

$$\mathbf{K}(C_{obs}) \cdot \theta = \mathbf{r}(C_{obs})$$

The vector $\mathbf{r}(C_{obs})$ represents the **"Fixed Residuals"**—the net rate of change in concentration for each group that occurs independently of the adjustable parameters. Calculating this analytically avoids the potential floating-point errors associated with large sparse matrix multiplications ($M_{const} \cdot C_{obs}$). 

---

## 1. Living Groups (Biota)
For a living group $i$, the residual represents the concentration lost to natural and fishing mortality. This loss must be balanced by variable gains (consumption) in the optimization.

$$r_i = Z_i \cdot C_{i, obs}$$

**Where:**
*   $Z_i$ is the total mortality rate: $Z_i = F_i + M2_i + M0_i + E_i$.
*   $F_i$: Fishing mortality.
*   $M2_i$: Predation mortality.
*   $M0_i$: Other mortality.
*   $E_i$: Emigration rate.

---

## 2. Detritus Groups
For detritus, the residual captures the net flow from fixed sources (mortality and discards) versus fixed sinks (consumption by others and export).

$$r_i = \left( \frac{\sum Q_{i,pred}}{B_i} + \text{Export}_i \right) C_{i, obs} - \sum_{j \in \text{Living}} \left( M0_j \cdot C_{j, obs} \cdot \text{Fate}_{j \to i} + \text{Discards}_{j \to i} \right)$$

**Key Terms:**
*   **Consumption Sink:** $\frac{\sum Q_{i,pred}}{B_i}$ is the rate at which detritus $i$ is consumed by all predators.
*   **Export Sink:** Physical export or burial rate of detritus.
*   **Mortality Source:** The concentration inflow from the non-predatory mortality ($M0$) of all living groups $j$, scaled by the fraction that goes to detritus group $i$.
*   **Discard Source:** Concentration inflow from fishery discards.

---

## 3. The Environment (Nutrients/Water)
The environment (index 0) acts as the ultimate sink for remineralized material. In the current formulation, detritus export is treated as a return to the environment.

$$r_0 = - \sum_{k \in \text{Detritus}} (\text{Export}_k \cdot C_{k, obs})$$

*Note: If the `base_inflow` is treated as a fixed constant, it is subtracted from this residual ($r_0 = \dots - \text{Inflow}$).*

---

## 4. Summary Table

| Component | Formula for $r_i$ | Physical Interpretation |
| :--- | :--- | :--- |
| **Living** | $Z_i \cdot C_i$ | Concentration lost to mortality. |
| **Detritus** | $(\text{Loss Rate}) C_i - (\text{Gain Flux})$ | Net imbalance from fixed flows. |
| **Environment**| $-\sum (\text{Export}_k \cdot C_k)$ | Total remineralization gain. |

---

## Benefits of Analytical Calculation
1.  **Precision:** Eliminates rounding errors from $M \cdot C$ operations involving near-zero coefficients.
2.  **Sparsity:** Direct calculation only iterates over non-zero biological links (diet and fate).
3.  **Stability:** All components are derived from positive physical rates, ensuring the residual scales linearly and predictably with observed concentrations.
