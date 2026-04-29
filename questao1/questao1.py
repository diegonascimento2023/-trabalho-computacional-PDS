# =============================================================
# Questão 1 - Algoritmo Recursivo para Cálculo de Raiz Quadrada
# =============================================================
# Disciplina: Processamento Digital de Sinais - UFERSA
# 
# O algoritmo calcula a raiz quadrada de um número A através
# da seguinte equação de recorrência:
#
#   y[n] = (1/2) * (y[n-1] + x[n] / y[n-1])
#
# onde x[n] = A*u[n] e y[-1] = A/2 é a estimativa inicial.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt

def calcular_raiz(A, N):
    """
    Calcula a raiz quadrada de A usando o algoritmo recursivo.
    
    Parâmetros:
    -----------
    A : float
        Número no qual deseja-se estimar a raiz quadrada
    N : int
        Número de iterações do algoritmo
    
    Retorna:
    --------
    y : numpy array
        Sequência y[n] com as estimativas da raiz quadrada
    erro : float
        Erro absoluto entre o valor real e o estimado
    """

    # Vetor para armazenar as estimativas y[n]
    y = np.zeros(N)

    # Estimativa inicial y[-1] = A/2
    y_anterior = A / 2

    # Sinal de entrada x[n] = A*u[n], ou seja, x[n] = A para n >= 0
    x = A

    # Calcula recursivamente y[n] para n = 0, 1, ..., N-1
    for n in range(N):
        y[n] = 0.5 * (y_anterior + x / y_anterior)
        y_anterior = y[n]

    # Valor real da raiz quadrada
    raiz_real = np.sqrt(A)

    # Erro absoluto entre o valor real e o estimado na última iteração
    erro = abs(raiz_real - y[N-1])

    return y, erro


def plotar_resultado(A, N, y, erro):
    """
    Plota o gráfico de y[n] e mostra o erro absoluto.
    
    Parâmetros:
    -----------
    A : float
        Número no qual deseja-se estimar a raiz quadrada
    N : int
        Número de iterações
    y : numpy array
        Sequência y[n] com as estimativas
    erro : float
        Erro absoluto final
    """

    # Eixo n (índices das amostras)
    n = np.arange(0, N)

    # Valor real da raiz quadrada (linha de referência)
    raiz_real = np.sqrt(A)

    plt.figure(figsize=(8, 4))

    # Plota as estimativas y[n]
    plt.stem(n, y, linefmt='blue', markerfmt='bo', basefmt='black',
             label=f'y[n] (estimativa)')

    # Plota a linha do valor real como referência
    plt.axhline(y=raiz_real, color='red', linestyle='--',
                label=f'√{A} = {raiz_real:.6f}')

    plt.title(f'Algoritmo Recursivo — A={A}, N={N}\nErro absoluto = {erro:.2e}')
    plt.xlabel('n (iterações)')
    plt.ylabel('y[n]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def executar_caso(A, N):
    """
    Executa e exibe os resultados para um caso específico.
    
    Parâmetros:
    -----------
    A : float
        Número no qual deseja-se estimar a raiz quadrada
    N : int
        Número de iterações
    """

    print(f'\n{"="*50}')
    print(f'A = {A}, N = {N}')
    print(f'Estimativa inicial: y[-1] = {A/2}')

    # Calcula a raiz quadrada recursivamente
    y, erro = calcular_raiz(A, N)

    # Exibe os resultados
    print(f'Valor real:     √{A} = {np.sqrt(A):.10f}')
    print(f'Valor estimado: y[{N-1}] = {y[N-1]:.10f}')
    print(f'Erro absoluto:  {erro:.2e}')

    # Plota o gráfico
    plotar_resultado(A, N, y, erro)


# =============================================================
# Casos solicitados no trabalho
# =============================================================

# (a) A = 5, N = 10
executar_caso(A=5, N=10)

# (b) A = 21, N = 25
executar_caso(A=21, N=25)

# (c) A = 21, N = 4
executar_caso(A=21, N=4)

# (d) A = 121, N = 30
executar_caso(A=121, N=30)