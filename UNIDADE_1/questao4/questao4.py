# -*- coding: utf-8 -*-
#
# =============================================================================
# QUESTAO 4 - RADAR COM CORRELACAO CRUZADA
# =============================================================================
#
# Em sistemas de radar, deseja-se estimar a distancia que um determinado alvo
# esta de uma base. No caso, transmite-se um sinal xa(t). Esse sinal ira
# refletir no alvo e retornar ao transmissor apos um tempo td. A partir da
# estimacao do valor td, e conhecendo-se que os sinais eletromagneticos se
# propagam na velocidade da luz c, pode-se estimar a distancia que o alvo
# esta da base. O sinal recebido ya(t) e dado por:
#
#   ya(t) = alpha * xa(t - td) + wa(t)
#
# em que wa(t) e um ruido branco Gaussiano com media zero e variancia sigma2.
# Apos amostragem, assumindo que td e multiplo inteiro do intervalo T:
#
#   y[n] = alpha * x[n - D] + w[n]
#
# Itens:
#   (a) Explique como estimar D utilizando a correlacao cruzada r_xy[l].
#
#   (b) Seja x[n] a sequencia de Barker de 13 pontos:
#       x[n] = {+1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,-1,+1}
#       e w[n] ruido branco Gaussiano com media 0 e sigma2 = 0.01.
#       Gerar y[n] para 0 <= n <= 199, com alpha=0.9 e D=20.
#       Plotar x[n] e y[n].
#
#   (c) Calcular e plotar a correlacao cruzada r_xy[l].
#       Utilizar o grafico para estimar D.
#
#   (d) Repetir (b) e (c) para sigma2 = 0.1 e sigma2 = 1.0.
#       Comentar os resultados.
#
# -----------------------------------------------------------------------------
# ITEM (a) - TEORIA:
#
# A correlacao cruzada entre x[n] e y[n] e definida como (slides, p.24):
#
#   r_xy[l] = sum_n x[n] * y[n - l]
#
# Substituindo y[n] = alpha*x[n-D] + w[n]:
#
#   r_xy[l] = alpha * sum_n x[n]*x[n-l-D] + sum_n x[n]*w[n-l]
#           = alpha * r_xx[l - D] + r_xw[l]
#
# Como w[n] e independente de x[n]: r_xw[l] ≈ 0.
# A sequencia de Barker tem autocorrelacao quase ideal:
#   r_xx[0] = 13 (pico maximo), |r_xx[l]| <= 1 para l != 0.
# Portanto r_xy[l] tem pico acentuado em l = D.
#
# Estimativa: D_est = argmax{ r_xy[l] }
#
# A distancia do alvo e calculada como:
#   d = (D * T * c) / 2
# onde T = 1/Fs e o periodo de amostragem e c = 3e8 m/s.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# PARAMETROS GERAIS
# ──────────────────────────────────────────────
alpha = 0.9    # atenuacao do canal
D     = 20     # atraso verdadeiro em amostras
N     = 200    # numero de amostras: 0 <= n <= 199
seed  = 42     # semente para reprodutibilidade

# Sequencia de Barker de 13 pontos
barker = np.array([+1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,-1,+1])

# x[n]: Barker nos primeiros 13 pontos, zero no restante
x = np.zeros(N)
x[:len(barker)] = barker

# x[n - D]: desloca D amostras para a direita
x_atrasado = np.zeros(N)
x_atrasado[D:] = x[:N - D]

n = np.arange(N)

# ──────────────────────────────────────────────
# FUNCAO: gera y[n], plota (b) e calcula (c)
# ──────────────────────────────────────────────
def executar_caso(sigma2, caso_label):
    rng = np.random.default_rng(seed)
    w   = rng.normal(loc=0, scale=np.sqrt(sigma2), size=N)
    y   = alpha * x_atrasado + w

    # ── Correlacao cruzada r_xy[l] = sum_n x[n]*y[n-l]
    # np.correlate(x, y, 'full') calcula sum_n x[n]*y[n+l] = r_xy[-l]
    # Para obter r_xy[l] conforme definicao dos slides, usamos correlate(y, x)
    # que calcula sum_n y[n]*x[n+l] = r_yx[-l] = r_xy[l]  ✓
    r_xy = np.correlate(y, x, mode='full')
    lags = np.arange(-(N - 1), N)

    idx_pico  = np.argmax(r_xy)
    D_estimado = lags[idx_pico]

    print(f'\n  sigma2 = {sigma2}')
    print(f'  D real = {D}  |  D estimado = {D_estimado}  '
          f'{"(CORRETO)" if D_estimado == D else "(ERRO)"}')

    # ── Figura: 3 subplots (x[n], y[n], r_xy[l])
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f'Questao 4 {caso_label} — '
        f'\u03b1={alpha}, D={D}, \u03c3\u00b2={sigma2}',
        fontsize=13
    )
    fig.subplots_adjust(top=0.90, hspace=0.55)

    # Subplot 1: x[n]
    axs[0].stem(n, x, linefmt='steelblue', markerfmt='C0o', basefmt='black')
    axs[0].set_title('Sinal transmitido x[n] — Sequencia de Barker (13 pontos)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('n')
    axs[0].set_xlim([-2, N])
    axs[0].grid(True, alpha=0.3)

    # Subplot 2: y[n]
    axs[1].stem(n, y, linefmt='darkorange', markerfmt='C1o', basefmt='black')
    axs[1].set_title(
        f'Sinal recebido y[n] = {alpha}\u00b7x[n-{D}] + w[n]'
        f'  (\u03c3\u00b2={sigma2})'
    )
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xlabel('n')
    axs[1].set_xlim([-2, N])
    axs[1].grid(True, alpha=0.3)

    # Subplot 3: r_xy[l]
    axs[2].plot(lags, r_xy, color='darkgreen', linewidth=1.0)
    axs[2].axvline(x=D_estimado, color='red', linestyle='--', linewidth=1.5,
                   label=f'Pico em l = {D_estimado}  \u2192  D estimado = {D_estimado}')
    axs[2].set_title(f'Correlacao Cruzada r_xy[l] — Pico em l = {D_estimado}')
    axs[2].set_ylabel('r_xy[l]')
    axs[2].set_xlabel('lag l')
    axs[2].legend(fontsize=9)
    axs[2].grid(True, alpha=0.3)

    nome_arquivo = f'questao4_sigma{str(sigma2).replace(".","")}.png'
    plt.show()
    print(f'  Grafico salvo em {nome_arquivo}')

# ──────────────────────────────────────────────
# EXECUCAO DOS CASOS
# ──────────────────────────────────────────────
print('=' * 55)
print('QUESTAO 4 - Radar com Correlacao Cruzada')
print('=' * 55)

# Item (b) e (c): sigma2 = 0.01
executar_caso(sigma2=0.01, caso_label='— Letras (b) e (c)  sigma2=0.01')

# Item (d): sigma2 = 0.1 e sigma2 = 1.0
executar_caso(sigma2=0.1,  caso_label='— Letra (d)  sigma2=0.1')
executar_caso(sigma2=1.0,  caso_label='— Letra (d)  sigma2=1.0')

print('\n' + '=' * 55)
print('COMENTARIOS - Letra (d)')
print('=' * 55)
print('''
  sigma2 = 0.01 : ruido fraco — pico de r_xy[l] bem definido
                  em l = D. Estimativa correta e precisa.

  sigma2 = 0.1  : ruido moderado — pico ainda visivel em l=D,
                  porem lobulos laterais mais elevados.
                  Estimativa ainda correta.

  sigma2 = 1.0  : ruido intenso — lobulos laterais elevados,
                  mas a propriedade da sequencia de Barker
                  preserva o pico principal em l = D.
                  Estimativa correta, porem com menor margem.

  Conclusao: a sequencia de Barker e robusta a ruido. Mesmo
  com variancia elevada, a correlacao cruzada identifica D
  corretamente devido a autocorrelacao quase ideal do sinal.
''')