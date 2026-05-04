# -*- coding: utf-8 -*-
# =============================================================
# Questao 4 - Radar com Correlacao Cruzada
# =============================================================
# Disciplina: Processamento Digital de Sinais - UFERSA
# Professor: Pedro Thiago Valerio de Souza
#
# TEORIA:
# Em sistemas de radar, um sinal x[n] e transmitido pela base.
# O sinal reflete no alvo e retorna atrasado de D amostras,
# com amplitude reduzida pelo fator alpha e contaminado por
# ruido branco gaussiano w[n]:
#
#   y[n] = alpha * x[n - D] + w[n]
#
# A sequencia de Barker de 13 pontos e utilizada como sinal
# transmitido por possuir autocorrelacao ideal: pico maximo
# em lag=0 e lobulos laterais de amplitude maxima igual a 1
# (contra 13 do pico). Isso garante deteccao precisa do atraso D.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# Parametros
# =============================================================
alpha  = 0.9       # atenuacao do canal
D      = 20        # atraso em amostras (verdadeiro)
sigma2 = 0.01      # variancia do ruido
N      = 200       # numero de amostras de y[n]: 0 <= n <= 199
seed   = 42        # semente para reprodutibilidade

# =============================================================
# Letra (b) - Geracao dos sinais x[n] e y[n]
# =============================================================
# Sequencia de Barker de 13 pontos
barker = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])

# x[n] definido para 0 <= n <= 199
# x[n] = barker[n] para 0 <= n <= 12, zero caso contrario
x = np.zeros(N)
x[:len(barker)] = barker

# y[n] = alpha * x[n - D] + w[n], para 0 <= n <= 199
# x[n - D] e obtido deslocando x[n] D amostras para a direita
# np.roll desloca circularmente, mas como x e zero apos n=12,
# o deslocamento linear e equivalente para n >= D
x_atrasado = np.zeros(N)
x_atrasado[D:] = x[:N - D]   # x[n - D]: desloca D amostras

# Ruido branco gaussiano: media=0, variancia=sigma2
rng = np.random.default_rng(seed)
w = rng.normal(loc=0, scale=np.sqrt(sigma2), size=N)

# Sinal recebido
y = alpha * x_atrasado + w

n = np.arange(N)

print('=' * 55)
print('LETRA (b) - Geracao dos sinais x[n] e y[n]')
print('=' * 55)
print(f'  Sequencia de Barker: {barker}')
print(f'  alpha  = {alpha}')
print(f'  D      = {D} amostras')
print(f'  sigma2 = {sigma2}')
print(f'  N      = {N} amostras')

# =============================================================
# Visualizacao - letra (b)
# =============================================================
fig, axs = plt.subplots(2, 1, figsize=(12, 7))
fig.subplots_adjust(top=0.93, hspace=0.45)

# Subplot 1: x[n]
axs[0].set_title('Questao 4 - Letra (b) — Sinais x[n] e y[n]\n'
                 'Sinal transmitido x[n] — Sequencia de Barker (13 pontos)')
axs[0].stem(n, x, linefmt='b', markerfmt='bo', basefmt='black')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('n')
axs[0].set_xlim([-2, N])
axs[0].grid(True, alpha=0.3)

# Subplot 2: y[n]
axs[1].stem(n, y, linefmt='r', markerfmt='ro', basefmt='black')
axs[1].set_title(f'Sinal recebido y[n] = {alpha}·x[n-{D}] + w[n]  '
                 f'(σ²={sigma2})')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlabel('n')
axs[1].set_xlim([-2, N])
axs[1].grid(True, alpha=0.3)

plt.show()