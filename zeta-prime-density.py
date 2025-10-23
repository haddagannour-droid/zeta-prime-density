import math
import csv
from math import log, pi
from sympy import isprime

def trouver_E_par_correction_fiable(n=10, k_max=20000, s=0.5+1j, epsilon=1e-4, max_iterations=50000):
    """
    Version fiable : retourne E seulement si la cible est atteinte.
    """
    print(f"  Précalcul des {k_max} termes...", end="", flush=True)
    termes_val = {}
    for k in range(1, k_max + 1):
        termes_val[k] = (n / k) ** s
    print(" OK")

    E = set()
    somme = 0j
    disponibles = set(range(1, k_max + 1))
    cible = -1 + 0j

    iteration = 0
    while disponibles and iteration < max_iterations:
        if abs(somme - cible) < epsilon:
            erreur_finale = abs(somme - cible)
            nb_premiers = sum(1 for k in E if isprime(k))
            return E, len(E), nb_premiers, erreur_finale, True

        meilleur_k = None
        meilleure_erreur = float('inf')
        for k in disponibles:
            erreur = abs(somme + termes_val[k] - cible)
            if erreur < meilleure_erreur:
                meilleure_erreur = erreur
                meilleur_k = k

        if meilleur_k is None:
            break

        E.add(meilleur_k)
        somme += termes_val[meilleur_k]
        disponibles.discard(meilleur_k)
        iteration += 1

    erreur_finale = abs(somme - cible)
    nb_premiers = sum(1 for k in E if isprime(k))
    return E, len(E), nb_premiers, erreur_finale, False


def zeros_non_triviaux_zeta(n=10):
    return [
        0.5 + 1j*14.13472514173469379045725198356247027078325070916076357842341945995273,
        0.5 + 1j*21.02203963877195263709580213189599257504534587088210678654670411279835,
        0.5 + 1j*25.01085758982382759792983208692416168675321983231093284781451544325917,
        0.5 + 1j*30.42487613757518013035425920828375320290393526106384336490126814741077,
        0.5 + 1j*32.96841867457137793610722856141930309681409631755527943707336279635764,
        0.5 + 1j*37.58617815941411226287720910929818065699279563241352721613262326870903,
        0.5 + 1j*40.91871901214716592444644977859308666240211829919561526652053178963421,
        0.5 + 1j*43.37793477537004967120119663512372437745359592355630242200294241650259,
        0.5 + 1j*48.00515088103236914836181189433121946136614879969559096105944019083133,
        0.5 + 1j*52.22703725070001318414600333688484648366487455414831018060652636619077
    ][:n]


def analyse_fiable():
    zeros = zeros_non_triviaux_zeta(10)
    # Ajuste k_max selon n
    parametres = [
        (20, 20000),
        (50, 20000),
        (100, 20000),
        (500, 50000),
        (1000, 50000),
        (2000, 100000)
    ]
    epsilon = 1e-4
    max_iterations = 50000

    lignes_csv = []
    resultats_valides = []

    print(f"{'='*90}")
    print(f"Analyse FIABLE avec affichage de la densité empirique")
    print(f"{'='*90}")

    for n, k_max in parametres:
        print(f"\n[Analyse pour n = {n:4d} | k_max = {k_max:6d}]")
        f_vals = []
        count_valid = 0

        for i, s in enumerate(zeros, 1):
            t = s.imag
            E, taille_E, nb_premiers, erreur_finale, convergé = trouver_E_par_correction_fiable(
                n=n, k_max=k_max, s=s, epsilon=epsilon, max_iterations=max_iterations
            )

            dens_emp = nb_premiers / taille_E if taille_E > 0 else 0
            print(f"  Zéro {i}: |E| = {taille_E:5d} | π(E) = {nb_premiers:3d} | densité = {dens_emp:.4f} | erreur = {erreur_finale:.2e} | convergé = {convergé}")

            if convergé and taille_E < k_max * 0.9:
                f_val = dens_emp * pi * log(n) if n > 1 else 0
                f_vals.append(f_val)
                count_valid += 1

                lignes_csv.append({
                    'n': n, 'zero': i, 't': t, 'f_nt': f_val,
                    'dens_emp': dens_emp, 'taille_E': taille_E, 'nb_premiers': nb_premiers,
                    'erreur_finale': erreur_finale, 'converge': True
                })
            else:
                lignes_csv.append({
                    'n': n, 'zero': i, 't': t, 'f_nt': None,
                    'dens_emp': dens_emp, 'taille_E': taille_E, 'nb_premiers': nb_premiers,
                    'erreur_finale': erreur_finale, 'converge': False
                })

        if f_vals:
            moyenne = sum(f_vals) / len(f_vals)
            ecart_type = math.sqrt(sum((x - moyenne)**2 for x in f_vals) / len(f_vals))
            resultats_valides.append((n, moyenne, ecart_type, len(f_vals)))
            print(f"  → Moyenne fiable f(n,t) = {moyenne:.3f} ± {ecart_type:.3f} (sur {len(f_vals)} zéros)")
        else:
            resultats_valides.append((n, None, None, 0))
            print(f"  → Aucun résultat convergé")

    # Résumé final
    print(f"\n{'='*90}")
    print(f"RÉSUMÉ DES RÉSULTATS CONVERGÉS")
    print(f"{'='*90}")
    print(f"{'n':>6} | {'Moyenne f':>10} | {'Écart-type':>12} | {'Zéros valides':>14}")
    print(f"{'-'*90}")
    for n, moy, std, count in resultats_valides:
        if moy is not None:
            print(f"{n:6} | {moy:10.3f} | {std:12.3f} | {count:14}")
        else:
            print(f"{n:6} | {'—':>10} | {'—':>12} | {count:14}")

    # Sauvegarde CSV
    with open('resultats_f_nt_fiables_avec_densite.csv', 'w', newline='') as csvfile:
        fieldnames = ['n', 'zero', 't', 'f_nt', 'dens_emp', 'taille_E', 'nb_premiers', 'erreur_finale', 'converge']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lignes_csv)
    print(f"\n✅ Résultats sauvegardés dans 'resultats_f_nt_fiables_avec_densite.csv'")

    return resultats_valides


if __name__ == "__main__":
    analyse_fiable()
