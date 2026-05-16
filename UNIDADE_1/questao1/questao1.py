# =============================================================
# Questão 1 - Algoritmo Recursivo para Cálculo de Raiz Quadrada
# =============================================================
# Disciplina: Processamento Digital de Sinais - UFERSA
#
# O algoritmo calcula a raiz quadrada de um número A através
# da seguinte equação de recorrência (Método de Newton-Raphson):
#
#   y[n] = (1/2) * (y[n-1] + x[n] / y[n-1])
#
# onde:
#   x[n] = A*u[n]  → entrada degrau de amplitude A
#   y[-1] = A/2    → estimativa inicial (condição inicial)
#   y[n]           → estimativa da raiz quadrada na iteração n
# =============================================================

import numpy as np
import matplotlib.pyplot as plt


def calcular_raiz(A, N):
    # Vetor que armazena y[n] para cada iteracao n = 0, 1, ..., N-1
    y = np.zeros(N)

    # Estimativa inicial: y[-1] = A/2 (condicao inicial do sistema)
    y_anterior = A / 2

    # Loop recursivo: cada y[n] depende de y[n-1]
    # Implementa a equacao: y[n] = 0.5 * (y[n-1] + A / y[n-1])
    for n in range(N):
        y[n] = 0.5 * (y_anterior + A / y_anterior)
        y_anterior = y[n]   # atualiza para a proxima iteracao

    # Erro absoluto: diferenca entre sqrt(A) real e a ultima estimativa
    erro = abs(np.sqrt(A) - y[N-1])

    return y, erro


def plotar(ax, A, N, letra):
    y, erro = calcular_raiz(A, N)
    raiz_real = np.sqrt(A)

    ax.plot(np.arange(N), y, color='steelblue', linewidth=1.5,
            marker='o', markersize=4, label='y[n] estimado')
    ax.axhline(y=raiz_real, color='red', linestyle='--', linewidth=1.5,
               label=f'√{A} = {raiz_real:.6f}')
    ax.set_title(f'({letra})  A={A},  N={N}  |  Erro = {erro:.2e}', fontsize=10)
    ax.set_xlabel('n (iteracoes)')
    ax.set_ylabel('y[n]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


fig, axs = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Questao 1 — Algoritmo Recursivo para Raiz Quadrada', fontsize=13)
fig.subplots_adjust(top=0.88, hspace=0.55, wspace=0.35)

plotar(axs[0, 0], A=5,   N=10, letra='a')
plotar(axs[0, 1], A=21,  N=25, letra='b')
plotar(axs[1, 0], A=21,  N=4,  letra='c')
plotar(axs[1, 1], A=121, N=30, letra='d')

plt.savefig('questao1.png', dpi=150, bbox_inches='tight')
plt.show()
print('Grafico salvo em questao1.png')
