# =============================================================
# Questão 2 - Filtro de Média Móvel
# =============================================================
# Disciplina: Processamento Digital de Sinais - UFERSA
#
# O filtro de média móvel é descrito pela equação:
#
#   y[n] = (1/N) * sum(x[n-k], k=0..N-1)
#
# onde N é a ordem do filtro.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

# =============================================================
# Item (a) e (b) - Resposta em frequência para N=3
# Magnitude e Fase
# =============================================================

def calcular_resposta_frequencia(N):
    """
    Calcula e plota a resposta em frequência do filtro
    de média móvel de ordem N.

    Parâmetros:
    -----------
    N : int
        Ordem do filtro de média móvel
    """

    # Coeficientes do filtro de média móvel
    # y[n] = (1/N) * (x[n] + x[n-1] + ... + x[n-N+1])
    # Numerador b = [1/N, 1/N, ..., 1/N] com N termos
    # Denominador a = [1] (sem realimentação)
    b = np.ones(N) / N
    a = [1.0]

    # Calcula a resposta em frequência usando o SciPy
    # w -> frequências em rad/amostra (de 0 a π)
    # H -> resposta em frequência complexa
    w, H = freqz(b, a, worN=1024)

    # Magnitude |H(e^jw)|
    magnitude = np.abs(H)

    # Fase ∠H(e^jw) em radianos
    fase = np.angle(H)

    # ---------------------------------------------------------
    # Plot da Magnitude
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(w, magnitude, 'b', linewidth=2)
    plt.title(f'Resposta em Frequência — Filtro Média Móvel N={N}')
    plt.ylabel('Magnitude |H(e^jω)|')
    plt.xlabel('Frequência (rad/amostra)')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['0', 'π/4', 'π/2', '3π/4', 'π'])
    plt.grid(True)
    plt.axhline(y=1/np.sqrt(2), color='r', linestyle='--',
                label='1/√2 (frequência de corte)')
    plt.legend()

    # ---------------------------------------------------------
    # Plot da Fase
    # ---------------------------------------------------------
    plt.subplot(2, 1, 2)
    plt.plot(w, fase, 'r', linewidth=2)
    plt.ylabel('Fase ∠H(e^jω) (rad)')
    plt.xlabel('Frequência (rad/amostra)')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['0', 'π/4', 'π/2', '3π/4', 'π'])
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Imprime os valores em algumas frequências importantes
    print(f'\n{"="*50}')
    print(f'Resposta em Frequência — N={N}')
    print(f'{"="*50}')
    print(f'ω=0:   |H| = {magnitude[0]:.4f}, ∠H = {fase[0]:.4f} rad')
    print(f'ω=π/2: |H| = {magnitude[256]:.4f}, ∠H = {fase[256]:.4f} rad')
    print(f'ω=π:   |H| = {magnitude[-1]:.4f}, ∠H = {fase[-1]:.4f} rad')

    # Identifica o tipo de filtro
    print(f'\nAnálise do tipo de filtro:')
    print(f'- Em ω=0 (baixa frequência):  |H| = {magnitude[0]:.4f} → passa')
    print(f'- Em ω=π (alta frequência):   |H| = {magnitude[-1]:.4f} → atenua')
    print(f'→ O filtro é do tipo PASSA-BAIXAS')

    return w, magnitude, fase


# =============================================================
# Item (c) - Implementação do filtro N=10 com SciPy
# Testa 3 variâncias diferentes
# =============================================================

def aplicar_filtro_media_movel(N, sigma2, n_amostras=51):
    """
    Aplica o filtro de média móvel de ordem N em um sinal
    senoidal contaminado por ruído Gaussiano.

    Parâmetros:
    -----------
    N : int
        Ordem do filtro de média móvel
    sigma2 : float
        Variância do ruído Gaussiano
    n_amostras : int
        Número de amostras (default=51, de n=0 até n=50)

    Retorna:
    --------
    n : array
        Índices das amostras
    r : array
        Sinal senoidal limpo
    eta : array
        Sinal de ruído
    x : array
        Sinal de entrada (r + ruído)
    y : array
        Sinal filtrado
    """

    # Vetor de índices n = 0, 1, ..., 50
    n = np.arange(n_amostras)

    # Sinal senoidal r[n] = sen(0.1*π*n)
    r = np.sin(0.1 * np.pi * n)

    # Ruído Gaussiano com média zero e variância sigma2
    # np.random.seed garante reprodutibilidade dos resultados
    np.random.seed(42)
    eta = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n_amostras)

    # Sinal de entrada x[n] = r[n] + η[n]
    x = r + eta

    # Coeficientes do filtro de média móvel de ordem N
    b = np.ones(N) / N
    a = [1.0]

    # Aplica o filtro usando lfilter do SciPy
    y = lfilter(b, a, x)

    return n, r, eta, x, y


def plotar_caso(N, sigma2):
    """
    Plota os 4 gráficos para um caso específico de N e sigma2.

    Parâmetros:
    -----------
    N : int
        Ordem do filtro
    sigma2 : float
        Variância do ruído
    """

    # Aplica o filtro
    n, r, eta, x, y = aplicar_filtro_media_movel(N, sigma2)

    # figsize maior para dar espaço ao suptitle
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))

    # subplots_adjust cria espaço manual:
    # top=0.93 → reserva 7% do topo para o suptitle
    # hspace=0.45 → espaço entre os subplots
    fig.subplots_adjust(top=0.93, hspace=0.45)

    fig.suptitle(f'Filtro Média Móvel N={N} — σ²={sigma2}', fontsize=14)

    # Gráfico 1 — Sinal senoidal r[n]
    axs[0].stem(n, r, linefmt='b', markerfmt='bo', basefmt='black')
    axs[0].set_title('(i) Sinal senoidal r[n] = sen(0.1πn)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # Gráfico 2 — Sinal de ruído η[n]
    axs[1].stem(n, eta, linefmt='r', markerfmt='ro', basefmt='black')
    axs[1].set_title(f'(ii) Ruído Gaussiano η[n] — σ²={sigma2}')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    # Gráfico 3 — Sinal de entrada x[n]
    axs[2].stem(n, x, linefmt='g', markerfmt='go', basefmt='black')
    axs[2].set_title('(iii) Sinal de entrada x[n] = r[n] + η[n]')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)

    # Gráfico 4 — Sinal filtrado y[n]
    axs[3].stem(n, y, linefmt='purple', markerfmt='o', basefmt='black')
    axs[3].plot(n, r, 'b--', label='r[n] original', alpha=0.5)
    axs[3].set_title('(iv) Sinal após filtragem y[n]')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_xlabel('n')
    axs[3].legend()
    axs[3].grid(True)

    plt.show()


def plotar_comparativo_variancias(N=10):
    """
    Plota um gráfico comparando os 3 casos de variância.

    Parâmetros:
    -----------
    N : int
        Ordem do filtro (default=10)
    """

    variancias = [0.01, 0.05, 0.1]
    cores = ['blue', 'orange', 'red']

    # figsize maior para dar espaço ao suptitle
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # subplots_adjust cria espaço manual:
    # top=0.90 → reserva 10% do topo para o suptitle
    # hspace=0.45 → espaço entre os subplots
    fig.subplots_adjust(top=0.90, hspace=0.45)

    fig.suptitle(f'Comparativo de Variâncias — Filtro Média Móvel N={N}',
                 fontsize=14)

    for i, sigma2 in enumerate(variancias):
        n, r, eta, x, y = aplicar_filtro_media_movel(N, sigma2)

        axs[i].plot(n, r, 'g--', label='r[n] original', alpha=0.7)
        axs[i].plot(n, y, color=cores[i],
                    label=f'y[n] filtrado σ²={sigma2}')
        axs[i].plot(n, x, 'gray', alpha=0.3, label='x[n] com ruído')
        axs[i].set_ylabel('Amplitude')
        axs[i].set_title(f'σ²={sigma2}')
        axs[i].legend()
        axs[i].grid(True)

    # xlabel apenas no último subplot
    axs[-1].set_xlabel('n')

    plt.show()


def comparar_ordens(sigma2=0.05):
    """
    Compara o desempenho do filtro para diferentes ordens.

    Parâmetros:
    -----------
    sigma2 : float
        Variância do ruído (default=0.05)
    """

    ordens = [5, 10, 50]
    cores = ['blue', 'orange', 'red']

    # figsize maior para dar espaço ao suptitle
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # subplots_adjust cria espaço manual:
    # top=0.90 → reserva 10% do topo para o suptitle
    # hspace=0.45 → espaço entre os subplots
    fig.subplots_adjust(top=0.90, hspace=0.45)

    fig.suptitle(f'Comparativo de Ordens — σ²={sigma2}',
                 fontsize=14)

    for i, N in enumerate(ordens):
        n, r, eta, x, y = aplicar_filtro_media_movel(N, sigma2)

        axs[i].plot(n, r, 'g--', label='r[n] original', alpha=0.7)
        axs[i].plot(n, y, color=cores[i], label=f'y[n] N={N}')
        axs[i].plot(n, x, 'gray', alpha=0.3, label='x[n] com ruído')
        axs[i].set_ylabel('Amplitude')
        axs[i].set_title(f'Ordem N={N}')
        axs[i].legend()
        axs[i].grid(True)

    # xlabel apenas no último subplot
    axs[-1].set_xlabel('n')

    plt.show()

# =============================================================
# Execução principal
# =============================================================

print('\n' + '='*50)
print('ITEM (a) e (b) - Resposta em Frequência N=3')
print('='*50)
calcular_resposta_frequencia(N=3)

print('\n' + '='*50)
print('ITEM (c) - Filtro N=10, testando variâncias')
print('='*50)

# Plota 4 gráficos para cada variância
for sigma2 in [0.01, 0.05, 0.1]:
    print(f'\nVariância σ²={sigma2}')
    plotar_caso(N=10, sigma2=sigma2)

# Plota gráfico comparativo das 3 variâncias
print('\nGráfico comparativo das variâncias...')
plotar_comparativo_variancias(N=10)

print('\n' + '='*50)
print('ITEM (e) - Comparando ordens N=5, N=10, N=50')
print('='*50)
comparar_ordens(sigma2=0.05)