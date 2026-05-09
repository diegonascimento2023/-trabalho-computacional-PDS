# Trabalho Computacional — Processamento Digital de Sinais

**Disciplina:** PEX0256 – Processamento Digital de Sinais  
**Docente:** Pedro Thiago Valério de Souza  
**Instituição:** UFERSA – Campus Pau dos Ferros  
**Semestre:** 2026.1  

---

## Estrutura do Repositório

```
/
├── UNIDADE_1/               ← 1º Trabalho Computacional
│   ├── questao1/
│   │   └── questao1.py      Algoritmo recursivo para raiz quadrada
│   ├── questao2/
│   │   └── questao2.py      Filtro de média móvel
│   ├── questao3/
│   │   ├── questao3.py      Detecção de complexo QRS no ECG
│   │   └── 100_norm.csv     Sinal ECG (MIT-BIH Arrhythmia Database)
│   ├── questao4/
│   │   └── questao4.py      Radar com correlação cruzada
│   ├── questao5/
│   │   └── questao5.py      Oscilador digital
│   └── questao6/
│       ├── questao6.py      Efeitos de áudio
│       └── guitar.wav       Áudio de guitarra para processamento
└── README.md
```

---

## UNIDADE 1 — Resumo das Questões

### Questão 1 — Algoritmo Recursivo para Raiz Quadrada
Implementação do algoritmo de Newton-Raphson discreto:

$$y[n] = \frac{1}{2}\left(y[n-1] + \frac{x[n]}{y[n-1]}\right), \quad x[n] = A \cdot u[n], \quad y[-1] = A/2$$

Testado para A=5 (N=10), A=21 (N=25), A=21 (N=4) e A=121 (N=30).

### Questão 2 — Filtro de Média Móvel
Análise analítica da resposta em frequência para N=3, classificação como passa-baixas e implementação com `scipy.signal.lfilter`. Testa variâncias de ruído (0.01, 0.05, 0.1) e ordens (N=5, 10, 50).

### Questão 3 — Detecção de Complexo QRS no ECG
Implementação do algoritmo de Ahlstrom & Tompkins (1983) baseado em primeira e segunda derivadas retificadas com detecção por duplo limiar (λ₁ = 0.5·max, λ₂ = 0.1·max) e período refratário de 200 ms. Aplicado ao dataset MIT-BIH Arrhythmia (Fs = 360 Hz, 650.000 amostras). Resultado: ~74.5 BPM.

### Questão 4 — Radar com Correlação Cruzada
Estimação do atraso D via pico da correlação cruzada r_xy[l], usando a sequência de Barker de 13 pontos como sinal transmitido. Testado com σ²=0.01, 0.1 e 1.0. D estimado = 20 (correto nos 3 casos).

### Questão 5 — Oscilador Digital
Oscilador baseado na equação de diferenças derivada de H(z) do cosseno (tabelado em Oppenheim). Gera cosseno, onda quadrada (comparador) e onda triangular (integrador retangular) para f₀=1kHz, Fs=10kHz.

### Questão 6 — Efeitos de Áudio
7 efeitos implementados sobre `guitar.wav` (Fs=44100 Hz, 6.18s): reversão temporal, subamostragem ×2, sobreamostragem ×2, delay, overdrive, tremolo e fuzz.

---

## Requisitos

```bash
pip install numpy matplotlib scipy soundfile
```

| Biblioteca | Uso |
|---|---|
| `numpy` | Operações vetoriais e geração de sinais |
| `matplotlib` | Visualização dos gráficos |
| `scipy` | Filtro de média móvel (`lfilter`) |
| `soundfile` | Leitura e escrita de arquivos `.wav` |

---

## Como Executar

Cada questão é executada de forma independente dentro de sua pasta:

```bash
cd UNIDADE_1/questao1
python questao1.py

cd UNIDADE_1/questao3
python questao3.py   # requer 100_norm.csv na mesma pasta

cd UNIDADE_1/questao6
python questao6.py   # requer guitar.wav na mesma pasta
```

---

## Observações Técnicas

- **Q3:** O enunciado sugere Fs=200 Hz, porém o professor autorizou usar a Fs detectada automaticamente pelo CSV. O dataset MIT-BIH usa Fs=360 Hz.
- **Q5:** O professor autorizou usar cosseno no lugar do seno — a transformada Z do cosseno é tabelada diretamente em Oppenheim, simplificando a dedução.
- **Q6 (Fuzz):** O expoente da distorção exponencial foi implementado como negativo (`-a·x²/|x|`) para garantir saturação física correta entre 0 e 1.
