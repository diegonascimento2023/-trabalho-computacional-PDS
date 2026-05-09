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
#       y[n] = x[n] + eta * x[n - Nd]
#       onde Nd = Fs * delta_t  (Nd: amostras de atraso)
#             eta: profundidade do efeito (|eta| < 1)
#
#   (e) Overdrive (distorcao suave por clipping definido por partes):
#       y[n] =  1                          se  x[n] >= 2/3
#       y[n] =  (3 - (2 - 3x[n])^2) / 3   se  1/3 <= x[n] < 2/3
#       y[n] =  2*x[n]                     se -1/3 <= x[n] < 1/3
#       y[n] = -(3 - (2 + 3x[n])^2) / 3   se -2/3 <= x[n] < -1/3
#       y[n] = -1                          se  x[n] < -2/3
#
#   (f) Tremolo (modulacao de amplitude):
#       y[n] = x[n] * (1 + eta * cos(2*pi*(Fx/Fs)*n))
#       Fx: frequencia do efeito em Hz (aprox. 5 Hz)
#       eta: profundidade do efeito (0 < eta < 1)
#
#   (g) Fuzz (distorcao exponencial com mix):
#       ye[n] = (x[n]/|x[n]|) * (1 - exp(-a * x[n]^2 / |x[n]|))
#       y[n]  = eta * ye[n] + (1 - eta) * x[n]
#       a: ganho (aprox. 10),  eta: profundidade (0 < eta < 1)
#       Obs.: o expoente e NEGATIVO para garantir saturacao entre
#       0 e 1 (convergencia fisica). O enunciado omite o sinal,
#       mas o expoente positivo divergiria para amplitudes altas.
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
eta_tremolo = 0.8    # profundidade do tremolo (0 < eta < 1)
Fx          = 5.0    # frequencia do tremolo (Hz)
eta_fuzz    = 0.8    # profundidade do fuzz (0 < eta < 1)
a_fuzz      = 10.0   # ganho do fuzz (aprox. 10)

# ──────────────────────────────────────────────
# (a) REVERSAO TEMPORAL
# y[n] = x[N - n + 1]
# ──────────────────────────────────────────────
y_a = x[::-1].copy()
sf.write('guitar_a_reverso.wav', y_a, Fs)
print('\n  (a) Reversao temporal gerada.')

# ──────────────────────────────────────────────
# (b) SUBAMOSTRAGEM x2 — pega 1 amostra a cada 2
# y[n] = x[2n]
# ──────────────────────────────────────────────
y_b = x[::2].copy()
sf.write('guitar_b_subamostrado.wav', y_b, Fs)
print(f'  (b) Subamostragem: {N} -> {len(y_b)} amostras.')

# ──────────────────────────────────────────────
# (c) SOBREAMOSTRAGEM x2 — insere zero entre amostras
# y[n] = x[n/2] para n par;  y[n] = 0 para n impar
# ──────────────────────────────────────────────
y_c = np.zeros(2 * N)
y_c[::2] = x
sf.write('guitar_c_sobreamostrado.wav', y_c, Fs)
print(f'  (c) Sobreamostragem: {N} -> {len(y_c)} amostras.')

# ──────────────────────────────────────────────
# (d) DELAY — eco com atraso de Nd amostras
# y[n] = x[n] + eta * x[n - Nd]
# ──────────────────────────────────────────────
Nd  = int(Fs * delta_t)    # Nd = Fs * delta_t
y_d = x.copy()
y_d[Nd:] = y_d[Nd:] + eta_delay * x[:N - Nd]
sf.write('guitar_d_delay.wav', y_d, Fs)
print(f'  (d) Delay: Nd={Nd} amostras ({delta_t}s), eta={eta_delay}.')

# ──────────────────────────────────────────────
# (e) OVERDRIVE — clipping suave definido por partes
# ──────────────────────────────────────────────
y_overdrive = np.where(x >= 2/3,   1.0,
              np.where(x >= 1/3,   (3 - (2 - 3*x)**2) / 3.0,
              np.where(x >= -1/3,  2*x,
              np.where(x >= -2/3, -(3 - (2 + 3*x)**2) / 3.0,
                                   -1.0))))
sf.write('guitar_e_overdrive.wav', y_overdrive, Fs)
print(f'  (e) Overdrive gerado.')

# ──────────────────────────────────────────────
# (f) TREMOLO — modulacao de amplitude
# y[n] = x[n] * (1 + eta*cos(2*pi*(Fx/Fs)*n))
# ──────────────────────────────────────────────
n_idx = np.arange(N)
y_f   = x * (1 + eta_tremolo * np.cos(2 * np.pi * (Fx / Fs) * n_idx))
sf.write('guitar_f_tremolo.wav', y_f, Fs)
print(f'  (f) Tremolo: Fx={Fx}Hz, eta={eta_tremolo}.')

# ──────────────────────────────────────────────
# (g) FUZZ — distorcao exponencial com mix
# ye[n] = (x[n]/|x[n]|) * (1 - exp(-a*x[n]^2/|x[n]|))
# y[n]  = eta*ye[n] + (1-eta)*x[n]
# ──────────────────────────────────────────────
eps   = 1e-10                          # evita divisao por zero
ax    = np.abs(x) + eps
sinal = x / ax                         # x[n]/|x[n]|: sinal de x
ye    = sinal * (1 - np.exp(-a_fuzz * x**2 / ax))   # distorcao
y_g   = eta_fuzz * ye + (1 - eta_fuzz) * x          # mix com original
sf.write('guitar_g_fuzz.wav', y_g, Fs)
print(f'  (g) Fuzz: a={a_fuzz}, eta={eta_fuzz}.')

# ──────────────────────────────────────────────
# GRAFICOS — janela de 1s para melhor visualizacao
# ──────────────────────────────────────────────
jan = int(1.0 * Fs)

sinais = [
    (x[:jan],              'Original x[n]',                                      'steelblue'),
    (y_a[:jan],            '(a) Reversao temporal',                              'slategray'),
    (y_b[:jan//2],         '(b) Subamostragem x2',                               'darkorange'),
    (y_c[:jan*2][:jan],    '(c) Sobreamostragem x2',                             'purple'),
    (y_d[:jan],            f'(d) Delay  (\u0394t={delta_t}s, \u03b7={eta_delay})', 'brown'),
    (y_overdrive[:jan],    '(e) Overdrive',                                      'crimson'),
    (y_f[:jan],            f'(f) Tremolo  (Fx={Fx}Hz, \u03b7={eta_tremolo})',    'teal'),
    (y_g[:jan],            f'(g) Fuzz  (a={a_fuzz}, \u03b7={eta_fuzz})',          'darkgreen'),
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

plt.savefig('questao6.png', dpi=150, bbox_inches='tight')
plt.show()
print('\n  Grafico salvo em questao6.png')
print('\n  Arquivos WAV gerados:')
for nome in ['guitar_a_reverso.wav', 'guitar_b_subamostrado.wav',
             'guitar_c_sobreamostrado.wav', 'guitar_d_delay.wav',
             'guitar_e_overdrive.wav', 'guitar_f_tremolo.wav',
             'guitar_g_fuzz.wav']:
    print(f'    {nome}')
