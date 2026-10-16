# Equação de Schrödinger e confinamento quântico

Trabalho do Tema 6 (Física Moderna) — Kauã e Eduardo Monteiro.

**Questão central:** como o confinamento de uma partícula modifica seus níveis de energia?

O programa resolve numericamente a equação de Schrödinger independente do tempo em 1D, por diferenças finitas (matriz Hamiltoniana tridiagonal), para:

- uma **caixa de potencial infinito** (comparada com a solução analítica Eₙ = n²π²ħ²/2mL²);
- um **poço de potencial finito** (estados ligados, penetração nas barreiras e transmissão por tunelamento).

Unidades: energia em eV e comprimento em nm. Partícula padrão: elétron livre.

## Como executar

```bash
pip install -r requirements.txt
python schrodinger_confinamento.py
```

Parâmetros opcionais:

```bash
python schrodinger_confinamento.py --larguras 1 2 4 --V0 1.0 --dx 0.005 --nestados 5
python schrodinger_confinamento.py --massa 0.067   # massa efetiva (em unidades de m_e)
```

| Parâmetro | Significado | Padrão |
|---|---|---|
| `--larguras` | larguras L do poço (nm) | 0.5 1.0 2.0 4.0 |
| `--V0` | altura das barreiras do poço finito (eV) | 1.0 |
| `--dx` | espaçamento da malha (nm) | 0.005 |
| `--nestados` | estados calculados na caixa infinita | 5 |
| `--barreira` | espessura da região de barreira de cada lado (nm) | 3.0 |
| `--massa` | massa em unidades de m_e | 1.0 |
| `--saida` | pasta de saída | resultados |

A execução imprime as tabelas no terminal e grava 9 figuras (PNG) e 3 tabelas (CSV) na pasta de saída.

## Executar no GitHub

- **Actions:** aba *Actions → Simulação → Run workflow*. Ao terminar, baixe o artefato `resultados`.
- **Codespaces:** botão *Code → Codespaces → Create codespace*, depois rode os comandos acima no terminal.

## Resultados obtidos (execução com os parâmetros padrão)

- Caixa infinita: erro relativo máximo de 0,205 % nas energias (L = 0,5 a 4 nm, n ≤ 5).
- Ordem de convergência medida: 2,000 (esperado: 2).
- Ajuste log-log de E₁ contra L: inclinação −1,99996 (esperado: −2).
- Poço finito (V₀ = 1 eV): 1, 2, 4 e 7 estados ligados para L = 0,5; 1; 2 e 4 nm.
- A transmissão T por tunelamento é calculada pela fórmula analítica de uma barreira retangular, não por simulação dinâmica.

A saída completa da execução está em `saida_execucao.txt`.

## Limitações

Modelo 1D, partícula única, massa constante, potencial de degraus abruptos, poço finito dentro de um domínio com ψ = 0 nas bordas. No poço finito o erro das energias cai proporcionalmente a dx (primeira ordem), por causa do degrau de potencial na malha.

## Referências

Consulte o relatório para a lista completa. Bibliotecas: NumPy, SciPy e Matplotlib.
