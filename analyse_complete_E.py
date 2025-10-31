import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
from collections import Counter

def premier_le_plus_proche(k):
    """Renvoie le nombre premier le plus proche de k."""
    if k < 2:
        return 2
    if isprime(k):
        return k
    # Recherche vers le bas
    d = 1
    while k - d >= 2:
        if isprime(k - d):
            p_inf = k - d
            break
        d += 1
    else:
        p_inf = None
    # Recherche vers le haut
    d = 1
    while True:
        if isprime(k + d):
            p_sup = k + d
            break
        d += 1
    if p_inf is None:
        return p_sup
    elif abs(k - p_inf) <= abs(k - p_sup):
        return p_inf
    else:
        return p_sup

def probabilite_proximite_premier(k, lambda_param=1.0):
    """Renvoie la probabilité de proximité première de k (version exponentielle)."""
    d = abs(k - premier_le_plus_proche(k))
    return np.exp(-lambda_param * d)

def construire_E_avec_analyse_complete(n=100, t=14.134725141734695, epsilon=1e-4, k_max=20000):
    s = 0.5 + 1j * t
    print(f"Construction de E pour n={n}, t={t:.6f}")
    
    # Précalcul des termes
    print("Précalcul des termes...", end="", flush=True)
    termes = {}
    for k in range(1, k_max + 1):
        termes[k] = (n / k) ** s
    print(" OK")

    E = set()
    somme = 0j
    disponibles = set(range(1, k_max + 1))
    cible = -1 + 0j
    iteration = 0
    max_iterations = 50000

    while disponibles and iteration < max_iterations:
        if abs(somme - cible) < epsilon:
            break
        meilleur_k = None
        meilleure_erreur = float('inf')
        for k in disponibles:
            erreur = abs(somme + termes[k] - cible)
            if erreur < meilleure_erreur:
                meilleure_erreur = erreur
                meilleur_k = k
        if meilleur_k is None:
            break
        E.add(meilleur_k)
        somme += termes[meilleur_k]
        disponibles.discard(meilleur_k)
        iteration += 1

    E = sorted(E)
    print(f"\n✅ Ensemble E construit : |E| = {len(E)}, erreur finale = {abs(somme + 1):.2e}")
    
    # === Affichage du plus petit et du plus grand k ===
    if E:
        min_k = min(E)
        max_k = max(E)
        print(f"  Plus petit k dans E : {min_k}")
        print(f"  Plus grand k dans E : {max_k}")
    else:
        print("  E est vide.")

    # Calcul des premiers proches et distances
    premiers_proches = []
    distances = []
    print("\n" + "="*60)
    print(f"{'k':>6} | {'Premier proche':>14} | {'Distance':>8}")
    print("="*60)
    for k in E:
        p = premier_le_plus_proche(k)
        d = abs(k - p)
        premiers_proches.append(p)
        distances.append(d)
        if len(distances) <= 20:  # Afficher les 20 premiers
            print(f"{k:6d} | {p:14d} | {d:8d}")
    if len(E) > 20:
        print("... (affichage tronqué)")

    # === Ensemble P : premiers associés ===
    P = set(premiers_proches)
    card_P = len(P)
    min_P = min(P) if P else None
    max_P = max(P) if P else None
    print(f"\n  Ensemble P (premiers associés à E) :")
    print(f"    Card(P) = {card_P}")
    if P:
        print(f"    Plus petit élément de P : {min_P}")
        print(f"    Plus grand élément de P : {max_P}")

    # === Densité des premiers proches ===
    densite_proche = card_P / len(E) if E else 0
    print(f"    Densité des premiers proches D_proche(E) = {densite_proche:.4f}")

    # === Comptage des k avec distance < 20 ===
    if distances:
        nb_k_inf_20 = sum(1 for d in distances if d < 20)
        pct_inf_20 = (nb_k_inf_20 / len(distances)) * 100
        print(f"\n  Nombre de k avec distance < 20 : {nb_k_inf_20} / {len(distances)} ({pct_inf_20:.2f}%)")

    # === Probabilité moyenne de proximité première ===
    probas = [probabilite_proximite_premier(k, lambda_param=1.0) for k in E]
    moyenne_proba = np.mean(probas)
    print(f"  Probabilité moyenne de proximité première : {moyenne_proba:.4f}")

    # Résumé statistique
    if distances:
        print("\n" + "="*60)
        print("Résumé statistique :")
        print(f"  Distance moyenne : {np.mean(distances):.2f}")
        print(f"  Distance médiane  : {int(np.median(distances))}")
        print(f"  Distance max     : {max(distances)}")
        nb_premiers_dans_E = sum(1 for k in E if isprime(k))
        print(f"  Premiers dans E  : {nb_premiers_dans_E} / {len(E)}")

    # === Pourcentages exacts par distance ===
    compteur = Counter(distances)
    total = len(distances)
    print("\n" + "="*50)
    print("Pourcentages par distance (exactes) :")
    print("="*50)
    for d in range(0, max(distances) + 1):
        freq = compteur.get(d, 0)
        if freq > 0:
            pct = (freq / total) * 100
            print(f"  Distance {d:2d} : {freq:4d} éléments → {pct:5.2f}%")

    # === GRAPHE 1 : k vs premier le plus proche ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(E, premiers_proches, color='red', s=20, alpha=0.6)
    plt.plot([min(E), max(E)], [min(E), max(E)], 'b--', linewidth=1, label='$y = x$')
    plt.xlabel('$k \\in E$', fontsize=11)
    plt.ylabel('Premier le plus proche', fontsize=11)
    plt.title('Premiers les plus proches', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # === GRAPHE 2 : Histogramme des distances ===
    plt.subplot(1, 2, 2)
    max_d = max(distances)
    plt.hist(distances, bins=range(0, max_d + 2), color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Distance au premier le plus proche', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title('Histogramme des distances', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(range(0, min(max_d + 1, 30)))

    plt.tight_layout()
    nom_base = f'E_analyse_n{n}_t{int(t)}'
    plt.savefig(f'{nom_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{nom_base}.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphes sauvegardés : {nom_base}.pdf et .png")
    plt.show()

# ======================
# Exécution
# ======================
if __name__ == "__main__":
    n = 100
    t = 14.134725141734695  # 1er zéro
    epsilon = 1e-4
    k_max = 20000

    construire_E_avec_analyse_complete(n=n, t=t, epsilon=epsilon, k_max=k_max)
