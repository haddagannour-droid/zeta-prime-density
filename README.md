# Prime Density in Additive Sets Associated with Nontrivial Zeros of the Riemann Zeta Function

This repository contains the **source code and data** supporting the article submitted to *Experimental Mathematics*.

The paper investigates the empirical density of prime numbers in finite additive sets \( E = E(n,t) \subset \mathbb{N} \) constructed to satisfy  
\[
\sum_{k \in E} \left( \frac{n}{k} \right)^{\frac{1}{2} + i t} \approx -1,
\]  
where \( s = \frac{1}{2} + i t \) is a nontrivial zero of the Riemann zeta function.

## 📁 Contents

- `ZetaPrimeDensity.tex`: LaTeX source of the manuscript (in English).
- `generer_graph.py`: Python script to reproduce the main figure (`f_vs_loglogn.pdf`).
- `analyse_complete_E.py`: Full script used to compute the arithmetic structure of \( E(n,t) \) (distances to primes, proximity probability, etc.).
- `f_vs_loglogn.pdf`: Final figure included in the manuscript.

## ▶️ How to Reproduce the Main Figure

To regenerate the key plot **Figure 1** (`f_vs_loglogn.pdf`), which shows the linear fit of  
\( f(n,t) = \pi \log n \cdot D_{\text{emp}}(n,t) \) versus \( \log \log n \),  
simply run:

```bash
python3 generer_graph.py
