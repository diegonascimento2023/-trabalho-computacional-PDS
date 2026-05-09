# -*- coding: utf-8 -*-
#
# =============================================================================
# QUESTAO 5 - OSCILADOR DIGITAL
# =============================================================================
#
# Muitos geradores de sinal utilizam tecnicas de processamento digital de
# sinais para a geracao de sinais analogicos. Uma das tecnicas consiste na
# criacao de um sinal senoidal digital que e posteriormente convertido em
# sinal analogico atraves de um conversor D/A. Uma forma simples para criar
# um oscilador digital consiste em criar um sistema cuja resposta ao impulso
# e o sinal senoidal desejado:
#
#   h[n] = A * sen(w0*n + phi) * u[n]
#
# sendo A a amplitude, w0 a frequencia angular em rad/amostra, phi a fase
# inicial e u[n] a funcao degrau unitario.
#
# Obs.: o professor autorizou usar cosseno no lugar do seno (mais simples,
#       pois a transformada Z e tabelada diretamente no livro de Oppenheim).
#
# Itens:
#   (a) Determinar H(z) e a equacao de diferencas do sistema.
#
#   (b) Determinar a expressao que relaciona w0 com Fs e f0. Calcular w0
#       para f0 = 1 kHz e Fs = 10 kHz.
#
#   (c) Implementar o sistema. Considerar A=1, phi=0, Fs=10 kHz, f0=1 kHz.
#       Gerar o sinal para 0 <= t <= 0.01 s e plotar.
#
#   (d) Gerar onda quadrada a partir da senoidal via comparador:
#       se h[n] >= 0 -> y=+1; caso contrario -> y=-1.
#
#   (e) Gerar onda dente de serra a partir da onda quadrada via integrador.
#       Aproximacao retangular (indicada pelo professor):
#           integral de 0 a t de xsq(tau)dtau ≈ T * sum_{k=0}^{n} xsq[k]
#       Em tempo discreto:
#           y_tri[n] = T * cumsum(y_sq[n])
#
# -----------------------------------------------------------------------------
# ITEM (a) - TEORIA:
#
# Usando cosseno: h[n] = cos(w0*n)*u[n]
#
# Transformada Z tabelada (Oppenheim):
#
#         1 - cos(w0)*z^-1
# H(z) = ─────────────────────────────
#         1 - 2*cos(w0)*z^-1 + z^-2
#
# Equacao de diferencas:
#
#   y[n] = 2*cos(w0)*y[n-1] - y[n-2] + x[n] - cos(w0)*x[n-1]
#
# Aplicando x[n] = delta[n] (impulso), a saida e y[n] = cos(w0*n).
#
# ITEM (b) - TEORIA:
#
# Relacao entre frequencia angular discreta e frequencias analogicas:
#
#   w0 = 2*pi*f0 / Fs
#
# Para f0 = 1 kHz e Fs = 10 kHz:
#
#   w0 = 2*pi*1000 / 10000 = 0.2*pi rad/amostra ≈ 0.6283 rad/amostra
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# PARAMETROS
# ──────────────────────────────────────────────
Fs  = 10000                     # frequencia de amostragem (Hz)
f0  = 1000                      # frequencia do sinal (Hz)
T   = 1.0 / Fs                  # periodo de amostragem (s)
w0  = 2 * np.pi * f0 / Fs      # frequencia angular discreta (rad/amostra)
N   = int(0.01 * Fs)            # 0 <= t <= 0.01 s -> 100 amostras
n   = np.arange(N)
t   = n * T                     # eixo de tempo em segundos

print('=' * 55)
print('QUESTAO 5 - Oscilador Digital')
print('=' * 55)
print(f'  Fs = {Fs} Hz')
print(f'  f0 = {f0} Hz')
print(f'  T  = {T*1e6:.1f} us')
print(f'  w0 = 2*pi*{f0}/{Fs} = 0.2*pi = {w0:.6f} rad/amostra')
print(f'  N  = {N} amostras  (0 <= t <= 0.01 s)')

# ──────────────────────────────────────────────
# ITEM (c) - OSCILADOR via equacao de diferencas
# y[n] = 2*cos(w0)*y[n-1] - y[n-2] + x[n] - cos(w0)*x[n-1]
# x[n] = delta[n]
# ──────────────────────────────────────────────
c = np.cos(w0)

x = np.zeros(N)
x[0] = 1.0          # impulso unitario

y_cos = np.zeros(N)
for i in range(N):
    xn   = x[i]
    xn_1 = x[i-1] if i >= 1 else 0.0
    yn_1 = y_cos[i-1] if i >= 1 else 0.0
    yn_2 = y_cos[i-2] if i >= 2 else 0.0
    y_cos[i] = 2*c*yn_1 - yn_2 + xn - c*xn_1

print(f'\n  Primeiras 5 amostras do oscilador: {np.round(y_cos[:5], 4)}')
print(f'  Verificacao cos(w0*n):             {np.round(np.cos(w0*n[:5]), 4)}')

# ──────────────────────────────────────────────
# ITEM (d) - ONDA QUADRADA via comparador
# ──────────────────────────────────────────────
y_sq = np.where(y_cos >= 0, 1.0, -1.0)

# ──────────────────────────────────────────────
# ITEM (e) - ONDA DENTE DE SERRA via integrador retangular
# y_tri[n] = T * cumsum(y_sq)  (normalizada para visualizacao)
# ──────────────────────────────────────────────
y_tri      = T * np.cumsum(y_sq)
y_tri_norm = y_tri / np.max(np.abs(y_tri))   # normaliza para [-1, +1]

# ──────────────────────────────────────────────
# GRAFICOS em tempo continuo (ms)
# ──────────────────────────────────────────────
t_ms = t * 1e3      # converte para milissegundos

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(
    'Questao 5 - Oscilador Digital  '
    '(Fs=10kHz, f0=1kHz, \u03c9\u2080=0.2\u03c0 rad/amostra)',
    fontsize=13
)
fig.subplots_adjust(top=0.90, hspace=0.55)

# Subplot 1: cosseno - letra (c)
axs[0].plot(t_ms, y_cos, color='steelblue', linewidth=1.5,
            marker='o', markersize=3)
axs[0].set_title('Letra (c) — Oscilador digital: h[n] = cos(\u03c9\u2080n)')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('Tempo (ms)')
axs[0].grid(True, alpha=0.3)

# Subplot 2: onda quadrada - letra (d)
axs[1].plot(t_ms, y_sq, color='darkorange', linewidth=1.5,
            marker='o', markersize=3)
axs[1].set_title('Letra (d) — Onda quadrada via comparador')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlabel('Tempo (ms)')
axs[1].set_ylim([-1.5, 1.5])
axs[1].grid(True, alpha=0.3)

# Subplot 3: dente de serra - letra (e)
axs[2].plot(t_ms, y_tri_norm, color='darkgreen', linewidth=1.5,
            marker='o', markersize=3)
axs[2].set_title('Letra (e) — Onda dente de serra via integrador retangular')
axs[2].set_ylabel('Amplitude (normalizada)')
axs[2].set_xlabel('Tempo (ms)')
axs[2].grid(True, alpha=0.3)

plt.show()
