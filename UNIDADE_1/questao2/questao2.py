# =============================================================
# Questão 2 - Filtro de Média Móvel
# =============================================================
# Disciplina: Processamento Digital de Sinais - UFERSA
#
# O filtro de média móvel é descrito pela equação de diferenças:
#
#   y[n] = (1/N) * sum_{k=0}^{N-1} x[n-k]
#
# Aplicando a DTFT e usando a propriedade de deslocamento temporal
# x[n-k] <-> e^{-jwk} X(e^{jw}), obtemos a função de transferência:
#
#   H(e^{jw}) = (1/N) * sum_{k=0}^{N-1} e^{-jwk}
#
# Para N=3, fatorando e^{-jw} para extrair a fase linear:
#
#   H(e^{jw}) = (1/3)(1 + e^{-jw} + e^{-j2w})
#             = (e^{-jw}/3)(e^{jw} + 1 + e^{-jw})
#             = (e^{-jw}/3)(1 + 2cos(w))
#
# Portanto:
#   |H(e^{jw})| = |1 + 2cos(w)| / 3     <- magnitude analítica
#   ∠H(e^{jw}) = -w                      <- fase linear analítica
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

# =============================================================
# Item (a) - Resposta em frequência ANALÍTICA para N=3
# =============================================================
# A expressão analítica foi obtida acima. Plotamos diretamente
# as fórmulas matemáticas, SEM usar freqz do SciPy.

print('=' * 55)
print('ITEM (a) - Resposta em Frequência Analítica (N=3)')
print('=' * 55)

# Vetor de frequências de 0 a π (1024 pontos)
w = np.linspace(0, np.pi, 1024)

# Magnitude analítica: |H(e^jw)| = |1 + 2*cos(w)| / 3
magnitude_analitica = np.abs(1 + 2 * np.cos(w)) / 3

# Fase analítica (conforme referência):
#   ∠H(e^jw) = -w,       se |w| < 2π/3  (termo real positivo)
#   ∠H(e^jw) = -w + π,   se |w| > 2π/3  (termo real negativo)
termo_real = 1 + 2 * np.cos(w)
correcao_fase = np.where(termo_real < 0, np.pi, 0)
fase_analitica = -w + correcao_fase

# Imprime valores em frequências-chave
print(f'\nValores analíticos em frequências-chave (N=3):')
print(f'  ω = 0:   |H| = {np.abs(1 + 2*np.cos(0))/3:.4f}')
print(f'  ω = π/2: |H| = {np.abs(1 + 2*np.cos(np.pi/2))/3:.4f}')
print(f'  ω = π:   |H| = {np.abs(1 + 2*np.cos(np.pi))/3:.4f}')

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(top=0.88, hspace=0.5)
fig.suptitle('Item (a) — Resposta em Frequência Analítica — Filtro Média Móvel N=3',
             fontsize=13)

# Magnitude
axs[0].plot(w, magnitude_analitica, 'b', linewidth=2)
axs[0].set_title(r'Magnitude: $|H(e^{j\omega})| = \frac{|1 + 2\cos\omega|}{3}$',
                 pad=10)
axs[0].set_ylabel(r'$|H(e^{j\omega})|$')
axs[0].set_xlabel('Frequência (rad/amostra)')
axs[0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
axs[0].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
axs[0].axhline(y=1/np.sqrt(2), color='r', linestyle='--',
               label=r'$1/\sqrt{2}$ ≈ 0.707 (referência -3dB)')
axs[0].legend()
axs[0].grid(True)

# Fase
axs[1].plot(w, fase_analitica, 'r', linewidth=2)
axs[1].set_title(r'Fase: $\angle H(e^{j\omega}) = -\omega + \arg(1 + 2\cos\omega)$')
axs[1].set_ylabel(r'$\angle H(e^{j\omega})$ (rad)')
axs[1].set_xlabel('Frequência (rad/amostra)')
axs[1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
axs[1].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
axs[1].set_yticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0])
axs[1].set_yticklabels(['-π', '-3π/4', '-π/2', '-π/4', '0'])
axs[1].grid(True)

plt.show()

# =============================================================
# Item (b) - Classificação do filtro
# =============================================================
# Analisando |H(e^jw)| = |1 + 2cos(w)| / 3:
#
#   ω = 0 (DC / baixas frequências): |H| = |1+2| / 3 = 1  → ganho máximo
#   ω = π (Nyquist / altas frequências): |H| = |1-2| / 3 = 1/3 → atenua
#
# O filtro preserva componentes de baixa frequência e atenua as
# de alta frequência → FILTRO PASSA-BAIXAS.

print('\n' + '=' * 55)
print('ITEM (b) - Classificação do Filtro')
print('=' * 55)
print(f'  ω = 0  (baixas freq.): |H| = {np.abs(1+2*1)/3:.4f} → PASSA')
print(f'  ω = π  (altas freq.):  |H| = {np.abs(1-2)/3:.4f}  → ATENUA')
print('  → Classificação: FILTRO PASSA-BAIXAS')

# =============================================================
# Item (c) - Implementação do filtro N=10 com SciPy
# Testa variâncias: 0.01, 0.05 e 0.1
# =============================================================
# A partir daqui usamos lfilter do SciPy para aplicar o filtro.
# Os coeficientes do filtro de média móvel de ordem N são:
#   b = [1/N, 1/N, ..., 1/N]  (N termos)
#   a = [1]                    (sem realimentação — filtro FIR)

print('\n' + '=' * 55)
print('ITEM (c) - Filtro N=10, testando variâncias')
print('=' * 55)


def aplicar_filtro_media_movel(N, sigma2, n_amostras=51, seed=42):
    """
    Aplica o filtro de média móvel de ordem N em um sinal
    senoidal contaminado por ruído Gaussiano Branco.

    Parâmetros
    ----------
    N        : ordem do filtro
    sigma2   : variância do ruído gaussiano
    n_amostras : número de amostras (padrão 51)
    seed     : semente para reprodutibilidade

    Retorna
    -------
    n, r, eta, x, y
    """
    n = np.arange(n_amostras)

    # Sinal senoidal: r[n] = sen(0.1·π·n)
    r = np.sin(0.1 * np.pi * n)

    # Ruído Gaussiano Branco: média=0, variância=sigma2
    rng = np.random.default_rng(seed)
    eta = rng.normal(loc=0, scale=np.sqrt(sigma2), size=n_amostras)

    # Sinal de entrada com ruído
    x = r + eta

    # Coeficientes FIR do filtro de média móvel
    b = np.ones(N) / N
    a = [1.0]

    # Filtragem via SciPy
    y = lfilter(b, a, x) # letra (a) mostra o que o filtro faz nas frequências, o lfilter executa esse filtro no tempo.

    return n, r, eta, x, y


def plotar_caso(N, sigma2):
    """Plota os 4 subgráficos para um par (N, sigma2)."""
    n, r, eta, x, y = aplicar_filtro_media_movel(N, sigma2)

    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    fig.subplots_adjust(top=0.93, hspace=0.45)
    fig.suptitle(f'Item (c) — Filtro Média Móvel  N={N},  σ²={sigma2}',
                 fontsize=13)

    # (i) Sinal senoidal limpo
    axs[0].stem(n, r, linefmt='b', markerfmt='bo', basefmt='black')
    axs[0].set_title('(i) Sinal senoidal  r[n] = sen(0.1πn)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # (ii) Ruído Gaussiano
    axs[1].stem(n, eta, linefmt='r', markerfmt='ro', basefmt='black')
    axs[1].set_title(f'(ii) Ruído Gaussiano Branco  η[n]  —  σ²={sigma2}')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    # (iii) Sinal com ruído
    axs[2].stem(n, x, linefmt='g', markerfmt='go', basefmt='black')
    axs[2].set_title('(iii) Sinal de entrada  x[n] = r[n] + η[n]')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)

    # (iv) Sinal filtrado vs. original
    axs[3].stem(n, y, linefmt='purple', markerfmt='o', basefmt='black')
    axs[3].plot(n, r, 'b--', alpha=0.6, label='r[n] original')
    axs[3].set_title('(iv) Sinal após filtragem  y[n]')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_xlabel('n')
    axs[3].legend()
    axs[3].grid(True)

    plt.show()


# Plota 4 gráficos para cada variância
for sigma2 in [0.01, 0.05, 0.1]:
    print(f'\n  Gerando gráficos para σ²={sigma2}...')
    plotar_caso(N=10, sigma2=sigma2)

# =============================================================
# Item (d) - Influência da variância do ruído
# =============================================================
# Gráfico comparativo das 3 variâncias no mesmo layout

print('\n' + '=' * 55)
print('ITEM (d) - Influência da variância do ruído')
print('=' * 55)
print('''
  σ² pequena (0.01): ruído de baixa amplitude → filtro
    praticamente reconstrói o sinal original r[n].

  σ² média (0.05): ruído moderado → filtro atenua bem,
    mas há leve distorção nas bordas (efeito transitório).

  σ² grande (0.1): ruído intenso → filtro ainda atenua,
    mas a saída apresenta maior desvio em relação a r[n].

  Conclusão: quanto maior a variância, maior a perturbação
  residual na saída — o filtro de média móvel tem desempenho
  degradado com ruídos de alta variância.
''')

variancias = [0.01, 0.05, 0.1]
cores = ['blue', 'orange', 'red']

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.subplots_adjust(top=0.90, hspace=0.45)
fig.suptitle('Item (d) — Influência da Variância do Ruído  (N=10)',
             fontsize=13)

for i, sigma2 in enumerate(variancias):
    n, r, eta, x, y = aplicar_filtro_media_movel(N=10, sigma2=sigma2)
    axs[i].plot(n, x, 'gray', alpha=0.35, label='x[n] com ruído')
    axs[i].plot(n, y, color=cores[i], linewidth=2,
                label=f'y[n] filtrado  σ²={sigma2}')
    axs[i].plot(n, r, 'g--', alpha=0.8, label='r[n] original')
    axs[i].set_title(f'σ²={sigma2}')
    axs[i].set_ylabel('Amplitude')
    axs[i].legend(fontsize=8)
    axs[i].grid(True)

axs[-1].set_xlabel('n')
plt.show()

# =============================================================
# Item (e) - Influência da ordem do filtro
# =============================================================
# Repete o item (c) para N=5 e N=50, comparando com N=10.

print('\n' + '=' * 55)
print('ITEM (e) - Comparativo de ordens (N=5, N=10, N=50)')
print('=' * 55)
print('''
  N pequena (N=5): janela curta → suavização moderada,
    responde rápido a variações, mas remove menos ruído.

  N média (N=10): equilíbrio entre suavização e fidelidade
    ao sinal original.

  N grande (N=50): janela longa → remove muito ruído, mas
    introduz atraso de grupo elevado e distorce transientes.
    As primeiras N-1 amostras da saída são afetadas pelo
    transitório do filtro (borda inicial com poucos dados).

  Conclusão: ordem maior → melhor supressão de ruído, porém
  maior atraso e distorção de borda. A escolha de N envolve
  compromisso entre suavização e fidelidade temporal.
''')

ordens = [5, 10, 50]
cores_e = ['blue', 'orange', 'red']

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.subplots_adjust(top=0.90, hspace=0.45)
fig.suptitle('Item (e) — Influência da Ordem do Filtro  (σ²=0.05)',
             fontsize=13)

for i, N in enumerate(ordens):
    n, r, eta, x, y = aplicar_filtro_media_movel(N=N, sigma2=0.05)
    axs[i].plot(n, x, 'gray', alpha=0.35, label='x[n] com ruído')
    axs[i].plot(n, y, color=cores_e[i], linewidth=2,
                label=f'y[n] filtrado  N={N}')
    axs[i].plot(n, r, 'g--', alpha=0.8, label='r[n] original')
    axs[i].set_title(f'Ordem N={N}')
    axs[i].set_ylabel('Amplitude')
    axs[i].legend(fontsize=8)
    axs[i].grid(True)

axs[-1].set_xlabel('n')
plt.show()