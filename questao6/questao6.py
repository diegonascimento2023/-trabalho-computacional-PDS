# -*- coding: utf-8 -*-
#
# =============================================================================
# QUESTAO 6 - EFEITOS DE AUDIO EM guitar.wav
# =============================================================================
#
# Sendo x[n] o sinal de audio original com N amostras nao nulas e y[n] o
# sinal de audio processado, implementar os seguintes efeitos de audio:
#
#   (a) Reversao temporal:
#       y[n] = x[N - n + 1],  0 <= n <= N-1
#
#   (b) Subamostragem por fator 2 (aumento de velocidade por 2):
#       y[n] = x[2n],  0 <= n <= (N-1)/2
#
#   (c) Sobreamostragem por fator 2 (diminuicao de velocidade por 2):
#       y[n] = x[n/2] se n = 0,2,4,...  ;  y[n] = 0 caso contrario
#
#   (d) Delay:
#       y[n] = x[n] + eta*x[n - Nd]
#       onde Nd = Fs * delta_t  (delta_t: atraso em segundos)
#
#   (e) Overdrive (distorcao suave por clipping):
#       y[n] = 2*x[n]                      se -1/3 <= x[n] < 1/3
#       y[n] = (3-(2-3x[n])^2)/3           se  1/3 <= x[n] < 2/3
#       y[n] = -(3-(2+3x[n])^2)/3          se -2/3 <= x[n] < -1/3
#       y[n] =  1                           se  x[n] >= 2/3
#       y[n] = -1                           se  x[n] < -2/3
#
#   (f) Tremolo (modulacao de amplitude):
#       y[n] = x[n] * (1 + eta*cos(2*pi*(Fx/Fs)*n))
#       onde Fx e a frequencia do efeito (aprox. 5 Hz)
#
#   (g) Fuzz (distorcao exponencial):
#       ye[n] = (x[n]/|x[n]|) * (1 - exp(a*x[n]^2/|x[n]|))
#       y[n]  = eta*ye[n] + (1-eta)*x[n]
#       onde a e o ganho (aprox. 10)
#
# Verificar os resultados de forma visual (plot no tempo) e auditiva.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# ──────────────────────────────────────────────
# LEITURA DO ARQUIVO
# ──────────────────────────────────────────────
x, Fs = sf.read('guitar.wav')

# Se estereo, usa apenas o canal esquerdo
if x.ndim > 1:
    x = x[:, 0]

N = len(x)
t = np.arange(N) / Fs

print('=' * 55)
print('QUESTAO 6 - Efeitos de Audio')
print('=' * 55)
print(f'  Arquivo : guitar.wav')
print(f'  Fs      : {Fs} Hz')
print(f'  Amostras: {N}')
print(f'  Duracao : {N/Fs:.2f} s')

# ──────────────────────────────────────────────
# PARAMETROS DOS EFEITOS
# ──────────────────────────────────────────────
eta_delay   = 0.5    # profundidade do delay
delta_t     = 0.3    # atraso do delay em segundos
eta_tremolo = 0.8    # profundidade do tremolo
Fx          = 5.0    # frequencia do tremolo (Hz)
eta_fuzz    = 0.8    # profundidade do fuzz
a_fuzz      = 10.0   # ganho do fuzz

# ──────────────────────────────────────────────
# (a) REVERSAO TEMPORAL
# ──────────────────────────────────────────────
y_a = x[::-1].copy()
sf.write('guitar_a_reverso.wav', y_a, Fs)
print('\n  (a) Reversao temporal gerada.')

# ──────────────────────────────────────────────
# (b) SUBAMOSTRAGEM x2 — velocidade 2x
# ──────────────────────────────────────────────
y_b = x[::2].copy()
sf.write('guitar_b_subamostrado.wav', y_b, Fs)
print(f'  (b) Subamostragem: {N} -> {len(y_b)} amostras.')

# ──────────────────────────────────────────────
# (c) SOBREAMOSTRAGEM x2 — velocidade 1/2
# ──────────────────────────────────────────────
y_c = np.zeros(2 * N)
y_c[::2] = x
sf.write('guitar_c_sobreamostrado.wav', y_c, Fs)
print(f'  (c) Sobreamostragem: {N} -> {len(y_c)} amostras.')

# ──────────────────────────────────────────────
# (d) DELAY (eco)
# ──────────────────────────────────────────────
Nd  = int(Fs * delta_t)
y_d = x.copy()
y_d[Nd:] = y_d[Nd:] + eta_delay * x[:N - Nd]
sf.write('guitar_d_delay.wav', y_d, Fs)
print(f'  (d) Delay: Nd={Nd} amostras ({delta_t}s), eta={eta_delay}.')

# ──────────────────────────────────────────────
# (e) OVERDRIVE — distorcao suave por clipping
# ──────────────────────────────────────────────
# Overdrive vetorizado conforme enunciado
y_e = np.where(x >= 2/3,   1.0,
      np.where(x >= 1/3,   (3 - (2 - 3*x)**2) / 3.0,
      np.where(x >= -1/3,  2*x,
      np.where(x >= -2/3, -(3 - (2 + 3*x)**2) / 3.0,
                           -1.0))))

y_overdrive = y_e.copy()
sf.write('guitar_e_overdrive.wav', y_overdrive, Fs)
print(f'  (e) Overdrive gerado.')

# ──────────────────────────────────────────────
# (f) TREMOLO — modulacao de amplitude
# ──────────────────────────────────────────────
n_idx = np.arange(N)
y_f   = x * (1 + eta_tremolo * np.cos(2 * np.pi * (Fx / Fs) * n_idx))
sf.write('guitar_f_tremolo.wav', y_f, Fs)
print(f'  (f) Tremolo: Fx={Fx}Hz, eta={eta_tremolo}.')

# ──────────────────────────────────────────────
# (g) FUZZ — distorcao exponencial
# ──────────────────────────────────────────────
# Evita divisao por zero: usa tiny onde x=0
eps   = 1e-10
ax    = np.abs(x) + eps
sinal = x / ax                                        # x[n]/|x[n]|
ye    = sinal * (1 - np.exp(-a_fuzz * x**2 / ax))    # expoente negativo
y_g   = eta_fuzz * ye + (1 - eta_fuzz) * x
sf.write('guitar_g_fuzz.wav', y_g, Fs)
print(f'  (g) Fuzz: a={a_fuzz}, eta={eta_fuzz}.')

# ──────────────────────────────────────────────
# GRAFICOS — janela de 1s para melhor visualizacao
# ──────────────────────────────────────────────
jan  = int(1.0 * Fs)   # 1 segundo de janela
t_j  = t[:jan]

sinais = [
    (x[:jan],          'Original x[n]',                          'steelblue'),
    (y_a[:jan],        '(a) Reversao temporal',                  'slategray'),
    (y_b[:jan//2],     '(b) Subamostragem x2',                   'darkorange'),
    (y_c[:jan*2:1][:jan], '(c) Sobreamostragem x2',              'purple'),
    (y_d[:jan],        f'(d) Delay  (\u0394t={delta_t}s, \u03b7={eta_delay})', 'brown'),
    (y_overdrive[:jan],'(e) Overdrive',                          'crimson'),
    (y_f[:jan],        f'(f) Tremolo  (Fx={Fx}Hz, \u03b7={eta_tremolo})', 'teal'),
    (y_g[:jan],        f'(g) Fuzz  (a={a_fuzz}, \u03b7={eta_fuzz})',       'darkgreen'),
]

fig, axs = plt.subplots(8, 1, figsize=(14, 26))
fig.suptitle('Questao 6 - Efeitos de Audio em guitar.wav', fontsize=14)
fig.subplots_adjust(top=0.93, bottom=0.04, hspace=1.0)

for i, (sinal, titulo, cor) in enumerate(sinais):
    t_plot = np.linspace(0, 1.0, len(sinal))
    axs[i].plot(t_plot, sinal, color=cor, linewidth=0.6)
    axs[i].set_title(titulo, fontsize=10, pad=4)
    axs[i].set_ylabel('Amp.')
    if i == 7:
        axs[i].set_xlabel('Tempo (s)')
    axs[i].grid(True, alpha=0.3)

plt.show()
print('\n  Grafico salvo em questao6.png')
print('\n  Arquivos WAV gerados:')
print('    guitar_a_reverso.wav')
print('    guitar_b_subamostrado.wav')
print('    guitar_c_sobreamostrado.wav')
print('    guitar_d_delay.wav')
print('    guitar_e_overdrive.wav')
print('    guitar_f_tremolo.wav')
print('    guitar_g_fuzz.wav')