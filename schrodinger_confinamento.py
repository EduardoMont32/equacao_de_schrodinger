#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equação de Schrödinger e confinamento quântico (Tema 6)
Equipe: Kauã e Eduardo Monteiro

Resolve numericamente a equação de Schrödinger independente do tempo 1D,

    -(hbar^2 / 2m) psi''(x) + V(x) psi(x) = E psi(x),

por diferenças finitas (matriz Hamiltoniana tridiagonal), para
  (1) caixa de potencial infinito de largura L;
  (2) poço de potencial finito (profundidade V0, largura L).

Unidades: energia em eV, comprimento em nm. Partícula: elétron livre (massa m_e).
Nessas unidades, C = hbar^2/(2 m_e) = 0,0380998 eV*nm^2.

Uso:
    python schrodinger_confinamento.py
    python schrodinger_confinamento.py --larguras 1 2 4 --V0 1.0 --dx 0.005 --nestados 5
    python schrodinger_confinamento.py --massa 0.067   # massa efetiva (GaAs), em unidades de m_e

Instalação:
    pip install numpy scipy matplotlib
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as sc
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Constantes físicas (SI, CODATA via scipy.constants) e conversão para eV e nm
# ----------------------------------------------------------------------------
HBAR = sc.hbar            # J s
M_E = sc.m_e              # kg
EV = sc.e                 # J por eV
NM = 1e-9                 # m por nm
# hbar^2/(2 m_e) em eV nm^2
C_FREE = HBAR**2 / (2.0 * M_E) / EV / NM**2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--larguras", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                   help="larguras L do poço, em nm")
    p.add_argument("--V0", type=float, default=1.0, help="altura das barreiras do poço finito, em eV")
    p.add_argument("--dx", type=float, default=0.005, help="espaçamento da malha, em nm")
    p.add_argument("--nestados", type=int, default=5, help="número de estados calculados (caixa infinita)")
    p.add_argument("--barreira", type=float, default=3.0, help="espessura da região de barreira de cada lado, nm")
    p.add_argument("--massa", type=float, default=1.0, help="massa da partícula em unidades de m_e")
    p.add_argument("--saida", type=str, default="resultados", help="pasta de saída")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Núcleo numérico
# ----------------------------------------------------------------------------
def resolver(x_int, V_int, C, selecao):
    """
    x_int : pontos INTERNOS da malha (psi=0 nas duas extremidades -> Dirichlet)
    V_int : potencial nesses pontos (eV)
    C     : hbar^2/(2m) em eV nm^2
    selecao : ('i', (i0, i1)) ou ('v', (Emin, Emax))

    Matriz H (tridiagonal):  H_ii = 2C/dx^2 + V_i ;  H_{i,i+1} = H_{i+1,i} = -C/dx^2
    Retorna energias (eV) e funções de onda normalizadas (sum |psi|^2 dx = 1) com psi=0 nas bordas.
    """
    dx = x_int[1] - x_int[0]
    t = C / dx**2
    d = 2.0 * t + V_int
    e = -t * np.ones(len(x_int) - 1)
    tipo, rng = selecao
    try:
        E, v = eigh_tridiagonal(d, e, select=tipo, select_range=rng)
    except Exception:
        return np.array([]), np.zeros((len(x_int) + 2, 0))
    # normalização: integral de |psi|^2 dx ~ soma |psi_i|^2 * dx = 1
    v = v / np.sqrt(np.sum(np.abs(v) ** 2, axis=0) * dx)
    # convenção de sinal: primeiro lobo significativo positivo (apenas estética)
    for j in range(v.shape[1]):
        idx = np.argmax(np.abs(v[:, j]) > 0.1 * np.max(np.abs(v[:, j])))
        if v[idx, j] < 0:
            v[:, j] *= -1
    psi = np.vstack([np.zeros((1, v.shape[1])), v, np.zeros((1, v.shape[1]))])
    return E, psi


def caixa_infinita(L, dx, nest, C):
    """Caixa infinita em [0, L]. Pontos internos x_i = i dx, i = 1..N, dx = L/(N+1)."""
    N = int(round(L / dx)) - 1
    h = L / (N + 1)
    x_int = h * np.arange(1, N + 1)
    E, psi = resolver(x_int, np.zeros(N), C, ("i", (0, nest - 1)))
    x = np.concatenate([[0.0], x_int, [L]])
    return x, E, psi


def energia_analitica_infinita(L, n, C):
    """E_n = n^2 pi^2 hbar^2 / (2 m L^2) = C (n pi / L)^2"""
    n = np.asarray(n, dtype=float)
    return C * (n * np.pi / L) ** 2


def pocos_finito(L, V0, dx, B, C):
    """
    Poço finito centrado em x=0: V=0 para |x|<L/2 e V=V0 fora.
    Domínio [-(L/2+B), +(L/2+B)] com psi=0 nas extremidades (B grande o bastante).
    Retorna x, V(x), energias ligadas (E<V0) e funções de onda.
    """
    W = L + 2.0 * B
    N = int(round(W / dx)) - 1
    h = W / (N + 1)
    x_int = -W / 2 + h * np.arange(1, N + 1)
    V_int = np.where(np.abs(x_int) < L / 2, 0.0, V0)
    E, psi = resolver(x_int, V_int, C, ("v", (-1e-9, V0)))
    x = np.concatenate([[-W / 2], x_int, [W / 2]])
    V = np.concatenate([[V0], V_int, [V0]])
    return x, V, E, psi


def energias_analiticas_finito(L, V0, C):
    """Equações transcendentais do poço finito (z = k L/2, z0 = (L/2) sqrt(V0/C)):
       par:    z tan z = sqrt(z0^2 - z^2);   ímpar: -z cot z = sqrt(z0^2 - z^2)
       E = C (2 z / L)^2.  Devolve lista ordenada de energias ligadas (eV)."""
    z0 = (L / 2.0) * np.sqrt(V0 / C)
    eps = 1e-9
    raizes = []
    m = 0
    while m * np.pi < z0:  # estados pares
        a, b = m * np.pi + eps, min(m * np.pi + np.pi / 2 - eps, z0)
        f = lambda z: z * np.tan(z) - np.sqrt(max(z0**2 - z**2, 0.0))
        if f(a) * f(b) < 0:
            raizes.append(brentq(f, a, b))
        m += 1
    m = 0
    while np.pi / 2 + m * np.pi < z0:  # estados ímpares
        a, b = np.pi / 2 + m * np.pi + eps, min((m + 1) * np.pi - eps, z0)
        g = lambda z: -z / np.tan(z) - np.sqrt(max(z0**2 - z**2, 0.0))
        if g(a) * g(b) < 0:
            raizes.append(brentq(g, a, b))
        m += 1
    z = np.sort(np.array(raizes))
    return C * (2.0 * z / L) ** 2, z0


def transmissao_barreira(E, V0, a, C):
    """T de uma barreira retangular (largura a, altura V0), E < V0:
       T = [1 + V0^2 sinh^2(kappa a) / (4 E (V0-E))]^-1, kappa = sqrt((V0-E)/C)."""
    kappa = np.sqrt((V0 - E) / C)
    return 1.0 / (1.0 + V0**2 * np.sinh(kappa * a) ** 2 / (4.0 * E * (V0 - E)))


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.saida, exist_ok=True)
    C = C_FREE / args.massa
    Ls = list(args.larguras)
    V0 = args.V0
    dx = args.dx
    ns = args.nestados
    cores = plt.cm.viridis(np.linspace(0.05, 0.85, len(Ls)))
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

    print("=" * 78)
    print(f"hbar^2/(2m) = {C:.6f} eV nm^2  (massa = {args.massa} m_e)")
    print(f"dx = {dx} nm ; V0 = {V0} eV ; larguras = {Ls} nm")
    print("=" * 78)

    # ------------------------- 1) CAIXA INFINITA ----------------------------
    print("\n### CAIXA INFINITA: energias numéricas x analíticas (eV)")
    linhas_inf = []
    dados_inf = {}
    for L in Ls:
        x, E, psi = caixa_infinita(L, dx, ns, C)
        n = np.arange(1, ns + 1)
        Ea = energia_analitica_infinita(L, n, C)
        err = np.abs(E - Ea) / Ea * 100
        dados_inf[L] = (x, E, psi, Ea)
        print(f"\nL = {L} nm  (N = {len(x)-2} pontos internos)")
        print(f"{'n':>3} {'E_num (eV)':>14} {'E_analit (eV)':>14} {'erro rel (%)':>14}")
        for i in range(ns):
            print(f"{n[i]:>3} {E[i]:>14.6f} {Ea[i]:>14.6f} {err[i]:>14.6f}")
            linhas_inf.append([L, n[i], E[i], Ea[i], err[i]])
    with open(os.path.join(args.saida, "tabela_caixa_infinita.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L_nm", "n", "E_num_eV", "E_analit_eV", "erro_rel_pct"])
        w.writerows(linhas_inf)

    # Verificação da lei E1 ~ 1/L^2: ajuste log-log
    Ls_fit = np.array(Ls)
    E1 = np.array([dados_inf[L][1][0] for L in Ls])
    slope, intercept = np.polyfit(np.log(Ls_fit), np.log(E1), 1)
    print(f"\nAjuste log-log E1 x L: inclinação = {slope:.5f} (esperado: -2)")
    print(f"E1*L^2 (eV nm^2) = {[round(E1[i]*Ls[i]**2, 5) for i in range(len(Ls))]}  (esperado: pi^2 C = {np.pi**2*C:.5f})")

    # ------------------------- 2) CONVERGÊNCIA ------------------------------
    print("\n### CONVERGÊNCIA (caixa infinita, L = 1 nm)")
    L_conv = 1.0
    Ns = [25, 50, 100, 200, 400, 800, 1600]
    conv = []
    print(f"{'N':>6} {'dx (nm)':>10} {'err E1 (%)':>14} {'err E2 (%)':>14} {'err E3 (%)':>14}")
    for N in Ns:
        h = L_conv / (N + 1)
        _, E, _ = caixa_infinita(L_conv, h, 3, C)
        # caixa_infinita arredonda L/dx - 1 = N
        Ea = energia_analitica_infinita(L_conv, np.arange(1, 4), C)
        er = np.abs(E - Ea) / Ea * 100
        conv.append([N, h, *er])
        print(f"{N:>6} {h:>10.5f} {er[0]:>14.3e} {er[1]:>14.3e} {er[2]:>14.3e}")
    conv = np.array(conv)
    p_ord = np.polyfit(np.log(conv[:, 1]), np.log(conv[:, 2]), 1)[0]
    print(f"Ordem de convergência observada (E1): {p_ord:.3f} (esperado: 2)")
    with open(os.path.join(args.saida, "tabela_convergencia.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "dx_nm", "erro_E1_pct", "erro_E2_pct", "erro_E3_pct"])
        w.writerows(conv.tolist())

    # ------------------------- 3) POÇO FINITO -------------------------------
    print(f"\n### POÇO FINITO (V0 = {V0} eV, barreiras de {args.barreira} nm de cada lado)")
    dados_fin = {}
    linhas_fin = []
    for L in Ls:
        x, V, E, psi = pocos_finito(L, V0, dx, args.barreira, C)
        Ea, z0 = energias_analiticas_finito(L, V0, C)
        nb_formula = int(np.ceil(z0 / (np.pi / 2)))
        dados_fin[L] = (x, V, E, psi, Ea)
        h = x[1] - x[0]
        print(f"\nL = {L} nm : z0 = {z0:.4f} ; estados ligados (numérico) = {len(E)} ; "
              f"(analítico, ceil(2 z0/pi)) = {nb_formula}")
        print(f"{'n':>3} {'E_num (eV)':>12} {'E_transc (eV)':>14} {'erro (%)':>10} "
              f"{'E_infinito (eV)':>16} {'P fora (%)':>11} {'1/kappa (nm)':>13}")
        for i in range(len(E)):
            pout = np.sum(np.abs(psi[np.abs(x) > L / 2, i]) ** 2) * h * 100
            kappa = np.sqrt((V0 - E[i]) / C)
            Ei = energia_analitica_infinita(L, i + 1, C)
            ea = Ea[i] if i < len(Ea) else np.nan
            er = abs(E[i] - ea) / ea * 100 if i < len(Ea) else np.nan
            print(f"{i+1:>3} {E[i]:>12.5f} {ea:>14.5f} {er:>10.4f} {Ei:>16.5f} {pout:>11.3f} {1/kappa:>13.4f}")
            linhas_fin.append([L, i + 1, E[i], ea, er, Ei, pout, 1 / kappa])
    with open(os.path.join(args.saida, "tabela_poco_finito.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L_nm", "n", "E_num_eV", "E_transc_eV", "erro_pct", "E_infinito_eV", "P_fora_pct", "penetracao_nm"])
        w.writerows(linhas_fin)

    # Número de estados ligados em função de L
    Lgrid = np.linspace(0.2, 6.0, 120)
    nb_curve = [int(np.ceil((L / 2 * np.sqrt(V0 / C)) / (np.pi / 2))) for L in Lgrid]

    # ------------------------- 4) TUNELAMENTO --------------------------------
    print("\n### TUNELAMENTO (barreira retangular, V0 = {:.2f} eV)".format(V0))
    a_grid = np.linspace(0.01, 2.0, 200)
    Es_t = [0.25 * V0, 0.5 * V0, 0.75 * V0]
    for a in [0.2, 0.5, 1.0]:
        print(f"a = {a} nm: " + ", ".join(f"T(E={E_:.2f} eV) = {transmissao_barreira(E_, V0, a, C):.3e}" for E_ in Es_t))

    # =========================== GRÁFICOS =====================================
    # Fig 1: V(x), níveis e funções de onda (caixa infinita) para cada largura
    fig, axs = plt.subplots(1, len(Ls), figsize=(4.0 * len(Ls), 4.6), squeeze=False)
    for ax, L in zip(axs[0], Ls):
        x, E, psi, Ea = dados_inf[L]
        esc = 0.35 * (E[1] - E[0]) if len(E) > 1 else 1.0
        for i in range(min(ns, 4)):
            ax.axhline(E[i], color="gray", lw=0.6, ls="--")
            ax.plot(x, E[i] + esc * psi[:, i] / np.max(np.abs(psi[:, i])), label=f"n={i+1}")
        ax.plot([0, 0, L, L], [E[3] * 1.2, 0, 0, E[3] * 1.2], "k", lw=2)
        ax.set_title(f"Caixa infinita, L = {L} nm")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Energia (eV)  [ψ deslocada só no gráfico]")
        ax.set_ylim(0, E[3] * 1.2)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig1_potencial_niveis_psi.png"))
    plt.close(fig)

    # Fig 2: ψ e |ψ|^2 para L de referência (a segunda largura da lista)
    Lref = Ls[1] if len(Ls) > 1 else Ls[0]
    x, E, psi, Ea = dados_inf[Lref]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(4):
        axs[0].plot(x, psi[:, i], label=f"ψ{i+1}, E={E[i]:.3f} eV")
        axs[1].plot(x, psi[:, i] ** 2, label=f"|ψ{i+1}|²")
    axs[0].set_title(f"Funções de onda (L = {Lref} nm)")
    axs[1].set_title(f"Densidades de probabilidade (L = {Lref} nm)")
    axs[0].set_ylabel("ψ(x)  (nm$^{-1/2}$)")
    axs[1].set_ylabel("|ψ(x)|²  (nm$^{-1}$)")
    for a_ in axs:
        a_.set_xlabel("x (nm)")
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig2_psi_e_densidade.png"))
    plt.close(fig)

    # Fig 3: níveis para diferentes larguras + E1 x L (log-log)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for k, L in enumerate(Ls):
        x, E, psi, Ea = dados_inf[L]
        for i in range(ns):
            axs[0].hlines(E[i], k - 0.35, k + 0.35, color=cores[k], lw=2)
            axs[0].text(k + 0.38, E[i], f"{i+1}", fontsize=7, va="center")
    axs[0].set_xticks(range(len(Ls)))
    axs[0].set_xticklabels([f"{L} nm" for L in Ls])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Energia (eV)  [escala log]")
    axs[0].set_xlabel("Largura L da caixa")
    axs[0].set_title("Níveis de energia (n = 1..{}) x largura".format(ns))
    Lc = np.linspace(min(Ls) * 0.8, max(Ls) * 1.2, 100)
    for n_ in range(1, 4):
        axs[1].loglog(Lc, energia_analitica_infinita(Lc, n_, C), "-", lw=1, label=f"analítico n={n_}")
        axs[1].loglog(Ls, [dados_inf[L][1][n_ - 1] for L in Ls], "o", ms=6)
    axs[1].set_xlabel("L (nm)")
    axs[1].set_ylabel("E$_n$ (eV)")
    axs[1].set_title(f"Lei $E_n \\propto L^{{-2}}$ (inclinação ajustada = {slope:.3f})")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig3_niveis_vs_largura.png"))
    plt.close(fig)

    # Fig 4: numérico x analítico e erro relativo
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for k, L in enumerate(Ls):
        x, E, psi, Ea = dados_inf[L]
        n = np.arange(1, ns + 1)
        axs[0].plot(n, E / Ea, "o-", color=cores[k], label=f"L = {L} nm")
        axs[1].semilogy(n, np.abs(E - Ea) / Ea * 100 + 1e-16, "o-", color=cores[k], label=f"L = {L} nm")
    axs[0].axhline(1, color="k", lw=0.7)
    axs[0].set_xlabel("n")
    axs[0].set_ylabel("E$_{num}$ / E$_{analit}$")
    axs[0].set_title("Razão numérico/analítico")
    axs[1].set_xlabel("n")
    axs[1].set_ylabel("Erro relativo (%)")
    axs[1].set_title(f"Erro relativo (dx = {dx} nm)")
    for a_ in axs:
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig4_num_vs_analitico.png"))
    plt.close(fig)

    # Fig 5: poço finito - V(x), níveis e ψ deslocadas
    fig, axs = plt.subplots(1, len(Ls), figsize=(4.0 * len(Ls), 4.6), squeeze=False)
    for ax, L in zip(axs[0], Ls):
        x, V, E, psi, Ea = dados_fin[L]
        ax.plot(x, V, "k", lw=2, label="V(x)")
        ax.axvspan(-L / 2, L / 2, color="tab:blue", alpha=0.07)
        esc = 0.08 * V0
        for i in range(len(E)):
            ax.axhline(E[i], color="gray", lw=0.5, ls="--")
            ax.plot(x, E[i] + esc * psi[:, i] / np.max(np.abs(psi[:, i])), lw=1.2)
        ax.set_xlim(-(L / 2 + 1.2), L / 2 + 1.2)
        ax.set_ylim(-0.03, V0 * 1.1)
        ax.set_title(f"Poço finito, L = {L} nm\n({len(E)} estado(s) ligado(s))")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Energia (eV)  [ψ deslocada só no gráfico]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig5_poco_finito.png"))
    plt.close(fig)

    # Fig 6: densidade nas barreiras (escala log) - penetração
    Lt = Ls[1] if len(Ls) > 1 else Ls[0]
    x, V, E, psi, Ea = dados_fin[Lt]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i in range(len(E)):
        axs[0].plot(x, psi[:, i] ** 2, label=f"n={i+1}, E={E[i]:.3f} eV")
        axs[1].semilogy(x, psi[:, i] ** 2 + 1e-30, label=f"n={i+1}")
    for a_ in axs:
        a_.axvspan(-Lt / 2, Lt / 2, color="tab:blue", alpha=0.07)
        a_.axvline(-Lt / 2, color="k", lw=0.8)
        a_.axvline(Lt / 2, color="k", lw=0.8)
        a_.set_xlabel("x (nm)")
        a_.set_xlim(-(Lt / 2 + 1.5), Lt / 2 + 1.5)
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    axs[0].set_ylabel("|ψ|² (nm$^{-1}$)")
    axs[1].set_ylabel("|ψ|² (nm$^{-1}$)  [escala log]")
    axs[1].set_ylim(1e-8, 10)
    axs[0].set_title(f"Densidade de probabilidade, L = {Lt} nm, V0 = {V0} eV")
    axs[1].set_title("Cauda evanescente nas barreiras (decaimento exponencial)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig6_penetracao_barreira.png"))
    plt.close(fig)

    # Fig 7: finito x infinito (E1..E3) e número de estados ligados
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for n_ in range(3):
        Lval, Ef, Ei = [], [], []
        for L in Ls:
            E = dados_fin[L][2]
            if len(E) > n_:
                Lval.append(L); Ef.append(E[n_]); Ei.append(energia_analitica_infinita(L, n_ + 1, C))
        axs[0].plot(Lval, Ef, "o-", label=f"finito n={n_+1}")
        axs[0].plot(Lval, Ei, "x--", color=axs[0].lines[-1].get_color(), label=f"infinito n={n_+1}")
    axs[0].axhline(V0, color="k", lw=0.8)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("L (nm)"); axs[0].set_ylabel("Energia (eV)")
    axs[0].set_title("Poço finito x caixa infinita")
    axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3, which="both")
    axs[1].step(Lgrid, nb_curve, where="post", label="analítico: ceil(2 z0/π)")
    axs[1].plot(Ls, [len(dados_fin[L][2]) for L in Ls], "ro", label="numérico")
    axs[1].set_xlabel("L (nm)"); axs[1].set_ylabel("Nº de estados ligados")
    axs[1].set_title(f"Estados ligados x largura (V0 = {V0} eV)")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig7_finito_vs_infinito.png"))
    plt.close(fig)

    # Fig 8: convergência
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for j, nome in enumerate(["E1", "E2", "E3"]):
        ax.loglog(conv[:, 1], conv[:, 2 + j], "o-", label=nome)
    ax.loglog(conv[:, 1], conv[0, 2] * (conv[:, 1] / conv[0, 1]) ** 2, "k--", lw=0.8, label="∝ dx²")
    ax.set_xlabel("dx (nm)"); ax.set_ylabel("Erro relativo (%)")
    ax.set_title(f"Convergência (L = 1 nm); ordem ≈ {p_ord:.2f}")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig8_convergencia.png"))
    plt.close(fig)

    # Fig 9: transmissão por tunelamento
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for E_ in Es_t:
        ax.semilogy(a_grid, transmissao_barreira(E_, V0, a_grid, C), label=f"E = {E_:.2f} eV")
    ax.set_xlabel("Largura da barreira a (nm)")
    ax.set_ylabel("Transmissão T")
    ax.set_title(f"Tunelamento: barreira retangular V0 = {V0} eV")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "fig9_tunelamento_T.png"))
    plt.close(fig)

    print(f"\nFiguras e tabelas CSV salvas em: {os.path.abspath(args.saida)}")


if __name__ == "__main__":
    main()
