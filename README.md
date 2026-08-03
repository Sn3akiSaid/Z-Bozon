# Z-Boson

Determining the mass, width and lifetime of the Z⁰ boson from electron–positron
collision data (e⁻e⁺ → Z⁰ → e⁻e⁺).

Coursework for an introductory physics Python course, originally submitted 2021 and
revisited for fun to fix some silly mistakes.

## What it does

Reads two CSV files of cross-section measurements, filters out invalid rows and
outliers, then fits the Breit–Wigner resonance

$$\sigma = \frac{12\pi}{m_Z^2}\cdot\frac{E^2\,\Gamma_{ee}^2}{(E^2-m_Z^2)^2+m_Z^2\Gamma_Z^2}$$

by minimising χ² over $m_Z$ and $\Gamma_Z$ simultaneously with `scipy.optimize.fmin`.
Uncertainties come from the χ²ᵐⁱⁿ + 1 contour.

## Results

| Quantity | Fitted | PDG |
| --- | --- | --- |
| Mass $m_Z$ | 91.18 ± 0.018 GeV/c² | 91.1876 |
| Width $\Gamma_Z$ | 2.509 ± 0.046 GeV | 2.4952 |
| Lifetime $\tau_Z$ | 2.623 × 10⁻²⁵ s | — |
| Reduced χ² | 0.947 | — |

Produces a fit plot over the data and a χ² contour plot showing the confidence regions.

## Running it

```bash
pip install numpy scipy matplotlib
python3 "Final Assignment Z- Bozon.py"
```

Expects `z_boson_data_1.csv` and `z_boson_data_2.csv` in the working directory.
