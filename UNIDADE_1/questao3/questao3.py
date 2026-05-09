# -*- coding: utf-8 -*-
#
# =============================================================================
# QUESTAO 3 - DETECCAO DO COMPLEXO QRS NO ECG
# =============================================================================
#
# O eletrocardiograma (ECG) e um sinal de importancia vital para o estudo dos
# fenomenos bio-eletricos. Em particular, o ECG contem informacoes para o
# diagnostico de doencas cardiovasculares, ja que esse reflete o comportamento
# biologico do coracao. Os sinais eletricos gerados pelo coracao podem ser
# capturados atraves de eletrodos posicionados no corpo do paciente. Cada
# trecho de um ciclo de eletrocardiograma foi classificado de acordo com o
# formato de cada onda e e dividido em onda P, complexo QRS, onda T e
# ocasionalmente onda U. A analise de um ECG e baseado em inspecao visual e
# identificacao do complexo QRS, que determina o inicio da contracao do
# ventriculo esquerdo. Todavia, em alguns casos, a analise visual do ECG e
# complexa, o que levou ao desenvolvimento de algoritmos computacionais que
# sao capazes de detectar o complexo QRS de um ECG.
#
# Dentre os algoritmos para deteccao do complexo QRS encontra-se o metodo de
# Ahlstrom e Tompkins (1983), baseado na primeira e segunda derivada.
# Sendo x[n] o sinal de ECG, no metodo de Ahlstrom e Tompkins primeiro
# calcula-se:
#
#   1a derivada retificada:
#       y0[n] = |x[n+1] - x[n-1]|
#
#   Suavizacao da 1a derivada retificada:
#       y1[n] = (y0[n-1] + 2*y0[n] + y0[n+1]) / 4
#
#   2a derivada retificada:
#       y2[n] = |x[n+2] - 2*x[n] + x[n-2]|
#
#   Combinacao final:
#       y3[n] = 2*(y1[n] + y2[n])
#
#   Limiares:
#       lambda1 = 0.5 * max{y3[n]}   (limiar primario)
#       lambda2 = 0.1 * max{y3[n]}   (limiar secundario)
#
#   Regra de deteccao:
#       - Percorre y3[n] ate encontrar amostra que excede lambda1
#         (candidato a QRS)
#       - As proximas 6 amostras consecutivas devem ser >= lambda2
#         para confirmacao do complexo QRS
#       - Apos cada deteccao, aplica periodo refratario de 200 ms
#         para evitar deteccoes duplas no mesmo complexo
#
#   De forma a minimizar erros devido ao inicio do processamento
#   (transientes resultantes do processo de captura), recomenda-se
#   descartar o primeiro 1.0 s do sinal ECG.
#
# Itens:
#   (a) Implementar o algoritmo em Python.
#       Obs.: o enunciado sugere Fs = 200 Hz, porem o professor
#       autorizou utilizar a Fs detectada automaticamente pelo CSV.
#       O arquivo 100_norm.csv pertence ao dataset MIT-BIH Arrhythmia,
#       cuja Fs real e 360 Hz — valor detectado pelo intervalo entre
#       amostras do eixo de tempo.
#
#   (b) Verificar o funcionamento do algoritmo com o sinal 100_norm.csv.
#       Apresentar em um mesmo grafico o sinal de ECG e a demarcacao
#       da posicao dos complexos QRS detectados.
#       Obs.: com 650.000 amostras, stem() e inviavel computacionalmente;
#       plot() e a representacao adequada para sinais longos.
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. LEITURA DO CSV
# ──────────────────────────────────────────────
# Formato: coluna 0 = tempo (s), coluna 1 = ECG canal 1, coluna 2 = ECG canal 2
dados = np.loadtxt('100_norm.csv', delimiter=',')
t   = dados[:, 0]
ecg = dados[:, 1]

# Fs detectada automaticamente pelo intervalo entre amostras
# MIT-BIH Arrhythmia Database: Fs = 360 Hz
diffs = np.diff(t[:200])
Fs = round(1.0 / np.mean(diffs))
print(f'Fs detectado: {Fs} Hz')
print(f'Duracao total: {t[-1]:.1f} s  |  Amostras: {len(ecg)}')

# ──────────────────────────────────────────────
# 2. DESCARTE DE TRANSIENTE INICIAL (1 s)
# ──────────────────────────────────────────────
descarte = int(1.0 * Fs)
t   = t[descarte:]
ecg = ecg[descarte:]
N   = len(ecg)

# ──────────────────────────────────────────────
# 3. ALGORITMO DE AHLSTROM & TOMPKINS
# ──────────────────────────────────────────────
# Implementacao vetorizada com numpy para eficiencia

# y0[n] = |x[n+1] - x[n-1]|   (1a derivada retificada)
# Realca regioes de mudanca rapida — bordas do complexo QRS
y0 = np.zeros(N)
y0[1:-1] = np.abs(ecg[2:] - ecg[:-2])

# y1[n] = (y0[n-1] + 2*y0[n] + y0[n+1]) / 4   (suavizacao)
# Remove ruido de alta frequencia mantendo os picos do QRS
y1 = np.zeros(N)
y1[1:-1] = (y0[:-2] + 2*y0[1:-1] + y0[2:]) / 4.0

# y2[n] = |x[n+2] - 2*x[n] + x[n-2]|   (2a derivada retificada)
# Detecta curvatura alta — QRS tem curvatura muito maior que ondas P e T
y2 = np.zeros(N)
y2[2:-2] = np.abs(ecg[4:] - 2*ecg[2:-2] + ecg[:-4])

# y3[n] = 2*(y1[n] + y2[n])   (combinacao final amplificada)
y3 = 2.0 * (y1 + y2)

# ──────────────────────────────────────────────
# 4. DETECCAO POR DUPLO LIMIAR
# ──────────────────────────────────────────────
lam1   = 0.5 * np.max(y3)    # limiar primario:   50% do maximo
lam2   = 0.1 * np.max(y3)    # limiar secundario: 10% do maximo
refrat = int(0.200 * Fs)      # periodo refratario: 200 ms em amostras

qrs_idx = []
i = 0
while i < N:
    if y3[i] >= lam1:
        # Candidato a QRS: verifica se as proximas 6 amostras >= lam2
        fim = min(i + 6, N)
        if np.all(y3[i:fim] >= lam2):
            qrs_idx.append(i)
            i += refrat    # pula periodo refratario
            continue
    i += 1

qrs_idx = np.array(qrs_idx)
t_qrs   = t[qrs_idx]

# Calculo da frequencia cardiaca media pelos intervalos RR
if len(qrs_idx) > 1:
    intervalos_RR = np.diff(t_qrs)
    bpm = 60.0 / np.mean(intervalos_RR)
    print(f'QRS detectados: {len(qrs_idx)}')
    print(f'FC media: {bpm:.1f} BPM')
else:
    print('Poucos QRS detectados - verificar limiares.')
    bpm = 0.0

# ──────────────────────────────────────────────
# 5. GRAFICOS
# ──────────────────────────────────────────────
# Janela de visualizacao: primeiros 10 s apos descarte
t0, t1  = t[0], t[0] + 10.0
mask    = (t >= t0) & (t <= t1)
t_win   = t[mask]
ecg_win = ecg[mask]
qrs_win = t_qrs[(t_qrs >= t0) & (t_qrs <= t1)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Questao 3 - Deteccao de Complexo QRS (Ahlstrom & Tompkins)',
             fontsize=13)
fig.subplots_adjust(top=0.88, hspace=0.55)

# ── Subplot 1: ECG + marcacoes QRS (item b)
ax1.plot(t_win, ecg_win, color='steelblue', linewidth=0.7, label='ECG canal 1')
for tq in qrs_win:
    ax1.axvline(x=tq, color='red', linewidth=1.0, alpha=0.8)
ax1.axvline(x=-999, color='red', linewidth=1.0, alpha=0.8,
            label=f'QRS detectado  (FC aprox. {bpm:.1f} BPM)')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude (mV)')
ax1.set_title(f'Sinal ECG com complexos QRS detectados  |  Fs = {Fs} Hz', fontsize=11)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(t0, t1)
ax1.grid(True, alpha=0.3)

# ── Subplot 2: zoom em torno de um QRS para visualizacao da morfologia
idx_zoom = 2       # 3o QRS detectado (evita bordas)
t_centro = qrs_win[idx_zoom]
margem   = 0.3     # janela de +/- 300 ms
tz0, tz1 = t_centro - margem, t_centro + margem
mask_z   = (t >= tz0) & (t <= tz1)

ax2.plot(t[mask_z], ecg[mask_z], color='steelblue', linewidth=1.2, label='ECG canal 1')
ax2.axvline(x=t_centro, color='red', linewidth=1.2, alpha=0.9, label='QRS detectado')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Amplitude (mV)')
ax2.set_title(f'Zoom no complexo QRS  (janela de {int(margem*1000)} ms antes/depois)', fontsize=11)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(tz0, tz1)
ax2.grid(True, alpha=0.3)

plt.savefig('questao3.png', dpi=150, bbox_inches='tight')
plt.show()
print('Grafico salvo em questao3.png')
