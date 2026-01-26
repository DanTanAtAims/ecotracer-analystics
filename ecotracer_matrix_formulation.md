# Linear System Formulation of Ecotracer Equilibrium

To analytically solve for the equilibrium state of the Ecotracer system, we define the mass-balance equations as a linear system of the form:

$$ \mathbf{M} \mathbf{x} = \mathbf{b} $$

Where:
*   $\mathbf{x}$ is the vector of unknown equilibrium states (concentrations and amounts).
*   $\mathbf{M}$ is the coefficient matrix representing transfer rates and losses.
*   $\mathbf{b}$ is the vector of independent forcing functions (external inflows).

## 1. State Vector ($\mathbf{x}$)

The state vector $\mathbf{x}$ has dimension $N+1$, where $N$ is the number of functional groups.

$$
\mathbf{x} = \begin{bmatrix}
C_{env} \\
A_1 \\
: \\
A_L \\
A_{L+1} \\
: \\
A_N
\end{bmatrix}
$$

*   Index $0$: $C_{env}$ (Environmental Concentration, e.g., $amount \cdot volume^{-1}$). 
*   Indices $1 \dots L$: $A_i$ (Total Contaminant Amount in **Living** groups $i$).
*   Indices $L+1 \dots N$: $A_k$ (Total Contaminant Amount in **Detritus** groups $k$).

> Note: $A_i = C_i \cdot B_i$, where $B_i$ is the biomass of group $i$.

---

## 2. The Linear System Matrix ($\mathbf{M}$)

The matrix $\mathbf{M}$ is constructed such that the diagonal elements represent total loss rates ($\beta$) and off-diagonal elements represent transfers (gains) between compartments.

$$ M_{ij} = \begin{cases} \beta_i & \text{if } i = j \text{ (Total Loss Rate)} \\ - \text{Transfer Rate}_{j \to i} & \text{if } i \neq j \text{ (Gain from } j \text{)} \end{cases} $$

### Row 0: Environment Dynamics

The environment loses contaminant via decay and uptake, and gains it from biota excretion and remineralization.

*   **Diagonal ($M_{0,0}$):** Total Loss Rate from Environment
    $$ \beta_{env} = k_{phys} + k_{vol\_exch} + \sum_{i=1}^{N} (u_i \cdot B_i) $$
    *   $k_{phys}$: Physical decay rate.
    *   $k_{vol\_exch}$: Loss due to volume exchange (flushing).
    *   $u_i \cdot B_i$: Direct uptake rate by group $i$ (Uptake parameter $\times$ Biomass).

*   **Off-Diagonal ($M_{0,j}$):** Gains from Group $j$ (Negative terms)
    $$ M_{0,j} = - \left( k_{meta, j} + k_{det\_out, j} + \sum_{p \in Predators} (1 - AE_p) \frac{Q_{pj}}{B_j} \right) $$
    *   $k_{meta, j}$: Metabolic decay/excretion rate of group $j$.
    *   $k_{det\_out, j}$: Remineralization rate from detritus $j$ to environment.
    *   **Unassimilated Excretion:** The sum represents unassimilated food returning to the water column.
        *   $Q_{pj}$: Consumption of prey $j$ by predator $p$.
        *   $AE_p$: Assimilation Efficiency of predator $p$.
        *   Dividing by $B_j$ converts the flux to be proportional to Amount $A_j$.

### Rows $1 \dots L$: Living Groups

Living groups lose contaminant via mortality and metabolism, and gain it from the environment and diet.

*   **Diagonal ($M_{i,i}$):** Net Loss Rate for Group $i$
    $$ M_{i,i} = \beta_i - \underbrace{\left( AE_i \cdot \frac{Q_{ii}}{B_i} \right)}_{\text{Cannibalism Correction}} $$
    *   $\beta_i = Z_i + k_{meta, i} + k_{phys, i}$ (Total Loss Rate).
    *   **Correction:** $Z_i$ includes mortality due to cannibalism ($M2_{ii} = Q_{ii}/B_i$). Since the group re-ingests this biomass, we subtract the assimilated gain from the total loss.
    *   *Result:* The net loss due to cannibalism is only the unassimilated fraction $(1 - AE_i) \cdot Q_{ii}/B_i$.

*   **Column 0 ($M_{i,0}$):** Direct Uptake from Environment
    $$ M_{i,0} = - (u_i \cdot B_i) $$

*   **Off-Diagonal ($M_{i,j}$ where $i \neq j$):** Dietary Uptake from Prey $j$
    $$ M_{i,j} = - \left( AE_i \cdot \frac{Q_{ij}}{B_j} \right) $$
    *   Gain is proportional to Consumption ($Q_{ij}$) scaled by prey concentration ($A_j / B_j$) and efficiency ($AE_i$). 

### Rows $L+1 \dots N$: Detritus Groups

Detritus accumulates contaminant from mortality and discards.

*   **Diagonal ($M_{k,k}$):** Total Loss Rate for Detritus $k$
    $$ \beta_k = \frac{\sum_p Q_{pk}}{B_k} + k_{phys, k} + k_{meta, k} + k_{det\_out, k} $$
    *   The first term represents loss due to consumption by scavengers/detritivores.

*   **Off-Diagonal ($M_{k,j}$):** Gains from Living Group $j$
    $$ M_{k,j} = - \left( (M0_j \cdot \text{Fate}_{j \to k}) + \sum_{f \in Fleets} \frac{D_{fj}}{B_j} \cdot \text{Fate}_{f \to k} \right) $$
    *   **Other Mortality:** $M0_j$ directs a portion of dead biomass (and contaminant) to detritus $k$.
    *   **Discards:** Fishery discards ($D_{fj}$) of group $j$ flow to detritus $k$.

---

## 3. Independent Vector ($\mathbf{b}$)

The vector $\mathbf{b}$ contains terms that are independent of the current state variables (constants).

$$ 
 b_i = \begin{cases}
I_{base} & \text{if } i = 0 \text{ (Environment)} \\
C_{immig, i} \cdot I_{immig, i} & \text{if } i > 0 \text{ (Biota/Detritus)}
\end{cases}
$$ 

*   $I_{base}$: Constant base inflow rate to the environment (e.g., runoff).
*   $C_{immig, i} \cdot I_{immig, i}$: Contaminant brought into the system by immigrating biomass of group $i$.

---

## 4. Solution

The equilibrium state is found by solving the linear system:

$$ \mathbf{x} = \mathbf{M}^{-1} \mathbf{b} $$

In practice (as implemented in `pyewe`), we use a numerical solver like `numpy.linalg.solve(M, b)` to handle cases where matrix inversion is computationally expensive or unstable.
