# genere_graphe.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Tes données fiables
n_vals = np.array([20, 50, 100, 500, 1000, 2000])
f_vals = np.array([0.956, 1.208, 1.511, 1.819, 2.028, 2.116])

x = np.log(np.log(n_vals))  # log log n

# Régression
slope, intercept, r_value, p_value, std_err = linregress(x, f_vals)
c, b = slope, intercept

# Points pour la droite
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = c * x_fit + b

# Tracer
plt.figure(figsize=(8, 6))
plt.scatter(x, f_vals, color='red', s=60, zorder=5, label='Données numériques')
plt.plot(x_fit, y_fit, 'b--', label=f'Régression: $R^2 = {r_value**2:.4f}$')

# Annoter les points
for i, n in enumerate(n_vals):
    plt.annotate(f'$n={n}$', (x[i], f_vals[i]), textcoords="offset points", xytext=(5,5))

plt.xlabel(r'$\log \log n$', fontsize=12)
plt.ylabel(r'$f(n,t) = \pi \log n \cdot D_{\mathrm{emp}}$', fontsize=12)
plt.title(r'Prime density scaling in additive sets linked to zeta zeros')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Sauvegarder
plt.savefig('f_vs_loglogn.pdf', dpi=300, bbox_inches='tight')
plt.savefig('f_vs_loglogn.png', dpi=300, bbox_inches='tight')
print("✅ Graphiques sauvegardés : f_vs_loglogn.pdf et .png")
