import numpy as np
import matplotlib.pyplot as plt

# Données expérimentales fiables (moyennes sur 10 zéros)
n_vals = np.array([20, 50, 100, 500, 1000, 2000])
f_vals = np.array([0.956, 1.208, 1.511, 1.819, 2.028, 2.116])

# Calcul de x = log(log(n))
x_vals = np.log(np.log(n_vals))

# Constantes théoriques
e = np.e
pi = np.pi
c_th = e / 2                      # ≈ 1.3591
b_th = -0.5 * np.sqrt(pi / e)    # ≈ -0.5375

# Points pour la droite théorique
x_th = np.linspace(x_vals.min() - 0.05, x_vals.max() + 0.05, 100)
y_th = c_th * x_th + b_th

# Tracé
plt.figure(figsize=(8, 6))
plt.scatter(x_vals, f_vals, color='red', s=80, zorder=5, label='Données numériques')
plt.plot(x_th, y_th, 'b--', linewidth=2, label=r'Th\'eorie : $\frac{e}{2} \log\log n - \frac{1}{2}\sqrt{\pi/e}$')

# Annotations des points
for i, n in enumerate(n_vals):
    plt.annotate(f'$n={n}$', (x_vals[i], f_vals[i]), textcoords="offset points", xytext=(5,5), fontsize=9)

# Style
plt.xlabel(r'$\log \log n$', fontsize=12)
plt.ylabel(r'$f(n,t) = \pi \log n \cdot D_{\mathrm{emp}}(n,t)$', fontsize=12)
plt.title(r'Prime density scaling and theoretical fit', fontsize=13)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Sauvegarde
plt.savefig('f_vs_loglogn.pdf', dpi=300, bbox_inches='tight')
plt.savefig('f_vs_loglogn.png', dpi=300, bbox_inches='tight')
print("✅ Graphes sauvegardés : f_vs_loglogn.pdf et .png")
plt.show()
