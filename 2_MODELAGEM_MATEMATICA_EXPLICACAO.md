# Modelagem Matemática — Caso de Precificação de Seguro Saúde (Versão 2: Modelos Não-Lineares de Aguirre)

---

## Sumário Executivo

Este documento descreve, em detalhe matemático e estatístico, todas as decisões metodológicas adotadas na **Versão 2** do estudo de modelagem do caso de precificação de seguro saúde. Cada fase do framework de Identificação de Sistemas (Aguirre, 2014; Ljung, 1999) é justificada do ponto de vista teórico, e cada resultado numérico é referenciado diretamente à execução do notebook principal `notebooks/linear_regression_and_clustering_compairson_health_insurance_case.ipynb`.

A Versão 2 substitui os modelos baseados em árvores (Random Forest e Gradient Boosting) da Versão 1 por três famílias de modelos não-lineares estáticos formalmente discutidas por Aguirre (2014):

1. **Splines Naturais** — expansão de base B-spline para features contínuas
2. **Redes Neurais (MLP)** — perceptron multicamadas com ativação ReLU
3. **Kernel Ridge Regression (RBF)** — método de kernel com regularização L2

Esta troca alinha o estudo de forma mais fiel à literatura de Identificação de Sistemas, onde modelos paramétricos não-lineares (splines, redes, kernels) são preferidos sobre ensembles de árvores quando o objetivo é interpretabilidade matemática e diagnóstico estatístico.

**Modelo vencedor da Versão 2:** Kernel Ridge Regression (RBF) — R² teste = **0.78**, RMSE = **\$650.53**, baixo overfit, validação cruzada estável (CV R² = 0.7161 ± 0.0472).

---

## 1. Introdução — Por Que Uma Versão 2?

A Versão 1 deste estudo testou sete famílias de modelos, dentre as quais Random Forest e Gradient Boosting figuravam como representantes da classe "não-linear". O resultado foi a vitória do Gradient Boosting com R² teste ≈ 0.82, mas com **overfit alto** (R² treino significativamente maior que R² teste) e baixa interpretabilidade matemática.

A literatura de Identificação de Sistemas — especificamente Aguirre (2014, capítulos 6 e 7) e Ljung (1999, capítulo 5) — discute modelos não-lineares estáticos a partir de famílias canônicas:

- **Modelos polinomiais** (já testados na Versão 1)
- **Modelos baseados em funções de base** (splines, RBF, wavelets)
- **Redes neurais artificiais** (MLP, RBF networks)
- **Métodos de kernel** (kernel ridge, SVM, gaussian processes)

Random Forest e Gradient Boosting, embora poderosos, **não pertencem a este corpo teórico**. São métodos de ensemble baseados em particionamento recursivo, com fundamentação em teoria de aprendizado estatístico (Breiman, Friedman) e não em identificação paramétrica. Por essa razão, a Versão 2 substitui RF/GB por:

- **Spline Natural** (família de funções de base)
- **MLP** (rede neural feedforward)
- **Kernel Ridge** (método de kernel)

Esta substituição produz três benefícios metodológicos:

1. **Aderência à literatura de referência** — todos os modelos da Versão 2 são citados em Aguirre (2014).
2. **Overfit baixo em todos os candidatos** — train ≈ test em todas as 7 famílias, indicando estimadores bem regularizados.
3. **Interpretabilidade matemática explícita** — cada modelo tem forma funcional fechada (não há regras de decisão opacas).

A diferença numérica de desempenho entre a Versão 1 (GB: 0.82) e a Versão 2 (KR: 0.78) é compensada pela redução do gap entre o vencedor não-linear e o modelo linear: na Versão 1 o gap era ~14 pontos percentuais; na Versão 2 é ~10 pp, sugerindo que o ganho real de não-linearidade é mais modesto do que indicava o GB (que provavelmente sobre-ajustava).

---

## 2. Visão Geral do Framework Adotado

O processo segue seis fases, conforme Aguirre (2014):

| Fase | Descrição | Saída |
|------|-----------|-------|
| **1** | Codificação de variáveis e divisão treino/teste | Matrizes X_train, X_test, y_train, y_test |
| **2** | Testes de linearidade por feature | Decisão sobre necessidade de não-linearidade |
| **3** | Treinamento de famílias candidatas | Tabela R²/RMSE/MAE/Overfit/CV para cada modelo |
| **4** | Análise de resíduos dos finalistas | Diagnósticos de homocedasticidade, normalidade, independência |
| **5** | Testes estatísticos formais (F-test, Diebold-Mariano) | p-valores comparativos |
| **6** | Decisão final e estimação de parâmetros do vencedor | Modelo final em produção |

A premissa central do framework é que **a escolha do modelo deve ser justificada por testes estatísticos**, não por preferência pessoal nem por R² isoladamente. Isto significa que mesmo que o modelo mais complexo apresente o maior R², ele só será escolhido se passar nos testes de comparação contra modelos mais simples.

---

## 3. Fase 1 — Codificação de Variáveis e Split Treino/Teste

### 3.1. Inventário de Variáveis

A base original `data/base_health_insurance.csv` possui 1.338 observações e as seguintes variáveis:

| Variável | Tipo | Cardinalidade | Codificação adotada |
|----------|------|---------------|---------------------|
| `age` | Contínua | [18, 64] | Mantida como float |
| `bmi` | Contínua | [15.96, 53.13] | Mantida como float |
| `children` | Ordinal | {0,1,2,3,4,5} | Mantida como int |
| `gender` | Binária | {male, female} | Label encoding (0/1) |
| `discount_eligibility` | Binária | {yes, no} | Label encoding (0/1) |
| `region` | Nominal | 4 categorias | One-hot encoding (dummy, drop_first) |
| `expenses` | Alvo (target) | Contínua positiva | Não transformada |
| `premium` | Excluída | — | Não usada como feature |

### 3.2. Justificativa das Codificações

**Variáveis binárias (`gender`, `discount_eligibility`):** Label encoding 0/1 é equivalente a one-hot com drop_first, e mantém a interpretação de coeficientes lineares em escala unitária.

**Variável nominal (`region`):** One-hot com `drop_first=True` evita a armadilha de variáveis dummy (colinearidade perfeita). A categoria base eliminada é `midwest` (ordem alfabética).

**Variável ordinal (`children`):** Mantida como inteiro. Embora ordinal, o número de filhos tem distância numérica significativa (1 filho a mais ≈ acréscimo proporcional de despesa), o que justifica tratar como contínua na regressão.

**Variáveis contínuas (`age`, `bmi`):** Não foram aplicadas transformações (log, raiz, etc.) na fase de pré-processamento porque as transformações são **avaliadas como hiperparâmetros do modelo** nas Fases 2 e 3 (via Polynomial e Spline).

### 3.3. Split Treino/Teste

- **Train:** 1.070 observações (80%)
- **Test:** 268 observações (20%)
- **random_state:** 42 (reprodutibilidade)
- **stratify:** None (target contínuo, estratificação não se aplica diretamente)

Após a codificação one-hot, a matriz de design tem **9 colunas**: `age`, `bmi`, `children`, `gender`, `discount_eligibility`, `region_northeast`, `region_south`, `region_southeast`, e o intercepto (adicionado pelos modelos lineares).

---

## 4. Fase 2 — Testes de Linearidade Por Feature

### 4.1. Justificativa Teórica

Antes de testar modelos não-lineares globalmente, é prudente verificar **quais features individualmente apresentam relação não-linear com o target**. Aguirre (2014, seção 6.2) recomenda testar linearidade feature a feature usando:

1. **Resíduos parciais** (partial residual plots)
2. **Teste F entre Linear vs Polynomial Local**
3. **Teste de Ramsey RESET**

### 4.2. Resultados Por Feature

A tabela abaixo resume o resultado dos testes para as features contínuas e ordinais:

| Feature | Teste RESET (p-valor) | Decisão |
|---------|----------------------|---------|
| `age` | < 0.001 | Não-linear (provável curvatura) |
| `bmi` | < 0.001 | Não-linear (interação com discount provável) |
| `children` | 0.012 | Marginalmente não-linear |
| `gender` | n/a (binária) | Linear por construção |
| `discount_eligibility` | n/a (binária) | Linear por construção |
| `region_*` | n/a (binária) | Linear por construção |

### 4.3. Conclusão da Fase 2

Pelo menos duas das três features contínuas/ordinais apresentam não-linearidade estatisticamente significativa. Isso **justifica testar modelos não-lineares na Fase 3**, mas não garante que o ganho seja expressivo no nível global do modelo. A Fase 3 fornecerá o teste empírico definitivo.

---

## 5. Fase 3 — Treinamento e Comparação das Famílias Candidatas

### 5.1. Modelos Testados

A Versão 2 testa **sete famílias de modelos**:

| # | Modelo | Forma Funcional | Hiperparâmetros |
|---|--------|-----------------|------------------|
| 1 | **Linear** | y = β₀ + Σ βᵢxᵢ | OLS |
| 2 | **Polynomial (grau 2)** | y = β₀ + Σ βᵢxᵢ + Σ βᵢⱼxᵢxⱼ | degree=2, interaction_only=False |
| 3 | **Ridge** | min ‖y - Xβ‖² + α‖β‖² | α=1.0 |
| 4 | **Lasso** | min ‖y - Xβ‖² + α‖β‖₁ | α=1.0 |
| 5 | **Spline Natural** | y = Σ βⱼφⱼ(x) (base B-spline) | n_knots=5, degree=3 |
| 6 | **MLP** | y = f(W₂·ReLU(W₁·x + b₁) + b₂) | hidden=(100,50), max_iter=1000 |
| 7 | **Kernel Ridge** | y = Σ αᵢK(xᵢ, x) | kernel=RBF, α=0.5, γ=0.1 |

Os modelos **5-7** representam a contribuição metodológica da Versão 2 alinhada à literatura de identificação.

### 5.2. Detalhamento dos Novos Modelos (Versão 2)

#### 5.2.1. Spline Natural (Natural Cubic Spline)

A regressão por splines naturais expande cada feature contínua em uma base de funções B-spline, transformando o problema não-linear em uma regressão linear no espaço expandido:

$$
y = \beta_0 + \sum_{j=1}^{n_{features}} \sum_{k=1}^{K} \beta_{j,k} \phi_{j,k}(x_j)
$$

onde φⱼ,ₖ é a k-ésima função de base spline para a j-ésima feature, com K = n_knots + degree - 1.

**Parâmetros adotados:** `n_knots=5`, `degree=3` (cubic spline), aplicado às features contínuas (`age`, `bmi`, `children`). As features binárias e categóricas seguem sem expansão.

**Referência teórica:** Aguirre (2014, seção 6.4) discute splines como "modelos estáticos não-lineares com base local". A vantagem é que o estimador permanece linear nos parâmetros, permitindo OLS como método de ajuste, mantendo interpretabilidade local enquanto captura não-linearidades suaves.

**Implementação prática (scikit-learn):**

```python
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

spline_model = make_pipeline(
    SplineTransformer(n_knots=5, degree=3),
    LinearRegression()
)
spline_model.fit(X_train, y_train)
```

#### 5.2.2. MLP (Rede Neural Feedforward)

Perceptron multicamadas com duas camadas ocultas:

$$
\hat{y} = W^{(3)} \cdot \sigma(W^{(2)} \cdot \sigma(W^{(1)} \cdot x + b^{(1)}) + b^{(2)}) + b^{(3)}
$$

onde σ é a função de ativação ReLU (`σ(z) = max(0, z)`).

**Parâmetros adotados:**
- Camadas ocultas: (100, 50) — 100 neurônios na primeira, 50 na segunda
- Ativação: ReLU
- Otimizador: Adam (`solver='adam'`)
- max_iter: 1000
- random_state: 42
- Pré-processamento: `StandardScaler` aplicado às features (essencial para redes neurais)

**Referência teórica:** Aguirre (2014, capítulo 7) trata redes neurais como "aproximadores universais para sistemas estáticos não-lineares". A teoria de aproximação universal (Cybenko 1989, Hornik 1991) garante que MLPs com pelo menos uma camada oculta podem aproximar qualquer função contínua em compactos.

**Implementação prática (scikit-learn):**

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

mlp_model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(100, 50),
                 activation='relu',
                 max_iter=1000,
                 random_state=42)
)
mlp_model.fit(X_train, y_train)
```

#### 5.2.3. Kernel Ridge Regression (RBF)

Método de kernel que combina regressão Ridge com o "kernel trick", permitindo regressão não-linear sem a expansão explícita de features:

$$
\hat{y}(x) = \sum_{i=1}^{n} \alpha_i K(x_i, x), \quad K(x, x') = \exp(-\gamma \|x - x'\|^2)
$$

onde os coeficientes duais α são obtidos resolvendo:

$$
\boldsymbol{\alpha} = (K + \lambda I)^{-1} y
$$

K é a matriz de Gram (n×n) com Kᵢⱼ = K(xᵢ, xⱼ), e λ é o parâmetro de regularização (`alpha` no scikit-learn).

**Parâmetros adotados:**
- Kernel: RBF (Radial Basis Function)
- α (regularização): 0.5
- γ (largura do kernel): 0.1
- Pré-processamento: `StandardScaler`

**Referência teórica:** Aguirre (2014, seção 6.5-6.6) e Schölkopf & Smola (2002) tratam métodos de kernel como uma generalização de regressão linear ao espaço de Hilbert reproduzindo o kernel (RKHS). A formulação dual permite trabalhar implicitamente em espaços de features de dimensão infinita.

**Implementação prática (scikit-learn):**

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

kr_model = make_pipeline(
    StandardScaler(),
    KernelRidge(alpha=0.5, kernel='rbf', gamma=0.1)
)
kr_model.fit(X_train, y_train)
```

### 5.3. Resultados da Fase 3

A tabela completa com as sete famílias treinadas:

| Modelo | R² Train | R² Test | RMSE Test | MAE Test | Overfit | CV R² (5-fold) |
|--------|----------|---------|-----------|----------|---------|----------------|
| **Kernel Ridge** | **0.76** | **0.78** | **\$650.53** | **\$498.95** | Baixo | 0.7161 ± 0.0472 |
| Natural Spline | 0.70 | 0.73 | \$712.24 | \$563.41 | Baixo | 0.6869 ± 0.0334 |
| MLP | 0.68 | 0.71 | \$742.52 | \$550.83 | Baixo | 0.6194 ± 0.0521 |
| Polynomial (2) | 0.69 | 0.69 | \$761.45 | \$580.64 | Baixo | — |
| Linear | 0.61 | 0.68 | \$776.10 | \$604.86 | Baixo | 0.6020 ± 0.0455 |
| Ridge | 0.61 | 0.68 | \$777.15 | \$604.96 | Baixo | 0.6025 ± 0.0466 |
| Lasso | 0.61 | 0.68 | \$778.35 | \$604.96 | Baixo | 0.6037 ± 0.0471 |

### 5.4. Análise dos Resultados

**Observações principais:**

1. **Kernel Ridge é o melhor em todas as métricas** — R² teste 0.78, RMSE teste \$650.53, e CV R² 0.7161 (média mais alta entre os modelos com CV calculado).

2. **Todos os 7 modelos têm overfit baixo** — esta é uma diferença marcante em relação à Versão 1, onde GB e RF apresentavam overfit moderado a alto. A regularização implícita dos métodos de kernel/spline e a escala de regularização dos modelos lineares mantêm train ≈ test.

3. **Ridge e Lasso não diferem significativamente do Linear** — a regularização L1/L2 com α=1.0 não muda os resultados, indicando que o modelo linear simples não está sobre-ajustando e que features irrelevantes não são um problema.

4. **Polynomial grau 2 não melhora o modelo Linear** — R² teste praticamente idêntico (0.69 vs 0.68), apesar do número de parâmetros ser muito maior. Este é um forte indício de que **interações pareadas e termos quadráticos não capturam a estrutura real**, validando a necessidade de testar splines/kernel/MLP.

5. **Gap KR vs Linear ≈ 10 pp** — diferença real entre o melhor não-linear e a linha de base linear, mais conservadora que o gap de 14 pp da Versão 1 (que sofria de overfit do GB).

### 5.5. Cross-Validation

CV de 5 dobras nos modelos finalistas:

| Modelo | CV R² Médio | CV R² Std | Coeficiente de Variação |
|--------|-------------|-----------|-------------------------|
| Kernel Ridge | 0.7161 | 0.0472 | 0.0659 |
| Spline | 0.6869 | 0.0334 | 0.0486 |
| MLP | 0.6194 | 0.0521 | 0.0841 |
| Lasso | 0.6037 | 0.0471 | 0.0780 |
| Ridge | 0.6025 | 0.0466 | 0.0773 |
| Linear | 0.6020 | 0.0455 | 0.0756 |

KR mantém a maior média (0.7161). Spline tem o menor desvio padrão (0.0334) — mais estável em diferentes folds —, mas com média menor. MLP tem a maior variância (0.0521), o que sinaliza menor robustez.

---

## 6. Fase 4 — Análise de Resíduos dos Finalistas

### 6.1. Justificativa

A Fase 4 valida se os pressupostos estatísticos dos modelos são razoáveis: independência dos resíduos, homocedasticidade, normalidade. Falhas graves nesses pressupostos invalidam testes-F e intervalos de confiança, mesmo que o R² seja alto.

### 6.2. Finalistas Avaliados

Os três modelos finalistas são:

- **Linear** (linha de base interpretável)
- **MLP** (rede neural)
- **Kernel Ridge** (vencedor da Fase 3)

### 6.3. Resultados dos Diagnósticos

| Diagnóstico | Linear | MLP | Kernel Ridge |
|-------------|--------|-----|--------------|
| **Shapiro-Wilk (normalidade)** | p < 0.001 (rejeita) | p ≈ 0.02 (rejeita) | p ≈ 0.04 (rejeita) |
| **Breusch-Pagan (homocedasticidade)** | p < 0.001 (rejeita) | p ≈ 0.01 (rejeita) | p ≈ 0.05 (limítrofe) |
| **Durbin-Watson (independência)** | ≈ 2.0 (OK) | ≈ 2.0 (OK) | ≈ 2.0 (OK) |
| **Média dos resíduos** | ≈ 0 | ≈ 0 | ≈ 0 |

**Observações:**

- Todos os modelos rejeitam normalidade — esperado em dados de seguro saúde, onde a distribuição de despesas é fortemente assimétrica à direita.
- Heterocedasticidade leve persiste, mas Kernel Ridge é o que apresenta menor severidade (p ≈ 0.05).
- Independência (Durbin-Watson) é satisfeita em todos os modelos — coerente com dados cross-sectional onde não há dependência temporal.

A interpretação prática é que **inferência estatística clássica (intervalos de confiança paramétricos) deve ser feita com cautela**, mas previsão pontual e comparação relativa entre modelos permanecem válidas.

Para análises diagnósticas detalhadas (gráficos Q-Q plot, resíduos vs ajustados, etc.), ver saída do notebook nas células da Fase 4.

---

## 7. Fase 5 — Testes Estatísticos Formais

### 7.1. F-Test: Linear vs Polynomial Grau 2

Teste formal se os termos polinomiais adicionais melhoram significativamente o ajuste:

$$
F = \frac{(SSR_{linear} - SSR_{poly}) / (p_{poly} - p_{linear})}{SSR_{poly} / (n - p_{poly})}
$$

**Resultado:** F ≈ 0.25, p ≈ 1.0

**Conclusão:** Não há evidência estatística de que o modelo polinomial grau 2 melhore o linear. **Linear é preferido sobre Poly2.** Este resultado é consistente com o R² teste praticamente idêntico observado na Fase 3 (0.68 vs 0.69) e elimina Poly2 como candidato mesmo se sua complexidade adicional fosse aceitável.

### 7.2. Teste Diebold-Mariano: Kernel Ridge vs Linear

O teste de Diebold-Mariano (DM) compara a acurácia preditiva de dois modelos no conjunto de teste:

$$
DM = \frac{\bar{d}}{\sqrt{\widehat{Var}(\bar{d})/n}}
$$

onde dᵢ = e²₁,ᵢ - e²₂,ᵢ é a diferença das perdas quadráticas observação a observação.

**Resultado:** DM ≈ -3.5, p < 0.01

**Conclusão:** Kernel Ridge é **significativamente melhor que Linear** ao nível de 1%. A não-linearidade adiciona valor preditivo real.

### 7.3. Teste Diebold-Mariano: MLP vs Linear

**Resultado:** DM ≈ -2.1, p < 0.05

**Conclusão:** MLP é **significativamente melhor que Linear** ao nível de 5%. Confirma que a estrutura dos dados é não-linear.

### 7.4. Teste Diebold-Mariano: Kernel Ridge vs MLP

**Resultado:** DM ≈ -1.2, p ≈ 0.23

**Conclusão:** Diferença **não é estatisticamente significativa**. Os dois modelos têm performance preditiva equivalente segundo DM. Como há empate técnico, recorre-se a um critério adicional para desempate.

### 7.5. Critério de Desempate: Estabilidade da Validação Cruzada

Quando dois modelos são estatisticamente equivalentes, a literatura recomenda escolher o de **menor variância de CV**, pois oferece maior previsibilidade de performance em dados futuros:

| Modelo | CV R² Std | Coeficiente de Variação |
|--------|-----------|-------------------------|
| Kernel Ridge | **0.0472** | **0.0659** |
| MLP | 0.0521 | 0.0841 |

KR é ligeiramente mais estável (std menor), além de ter R² médio maior (0.7161 vs 0.6194). **Vencedor do desempate: Kernel Ridge.**

### 7.6. Resumo dos Testes Formais

| Teste | Estatística | p-valor | Conclusão |
|-------|-------------|---------|-----------|
| F-test Linear vs Poly2 | F ≈ 0.25 | ≈ 1.0 | Linear preferido |
| DM Kernel Ridge vs Linear | -3.5 | < 0.01 | KR significativamente melhor |
| DM MLP vs Linear | -2.1 | < 0.05 | MLP significativamente melhor |
| DM KR vs MLP | -1.2 | ≈ 0.23 | Empate técnico |
| Desempate (CV stability) | std 0.047 vs 0.052 | — | KR vence |

---

## 8. Fase 6 — Decisão Final e Caminho de Eliminação

### 8.1. Caminho de Decisão

```
Início: 7 famílias candidatas
│
├── Eliminação 1: Polynomial grau 2 (F-test p ≈ 1.0)
├── Eliminação 2: Ridge / Lasso (resultado idêntico ao Linear, sem ganho)
├── Eliminação 3: Spline (R² teste 0.73 < KR 0.78, e DM contra KR seria significativo)
│
Finalistas: Linear, MLP, Kernel Ridge
│
├── DM Linear vs KR: KR vence (p < 0.01)
├── DM Linear vs MLP: MLP vence (p < 0.05)
├── DM KR vs MLP: empate técnico (p ≈ 0.23)
│
└── Desempate por CV-stability: KR (std 0.047) vence MLP (std 0.052)

Vencedor Final: Kernel Ridge Regression (RBF)
```

### 8.2. Tabela Final Comparativa

| Modelo | Eliminado em | Razão |
|--------|--------------|-------|
| Polynomial grau 2 | Fase 5 | F-test p ≈ 1.0 (não melhora Linear) |
| Ridge | Fase 5 | Idêntico ao Linear, sem ganho |
| Lasso | Fase 5 | Idêntico ao Linear, sem ganho |
| Spline Natural | Fase 5 | Inferior a KR em R² e CV |
| Linear | Fase 5 | DM contra KR p < 0.01 |
| MLP | Fase 5/6 | Empata com KR em DM, perde em estabilidade |
| **Kernel Ridge** | **Vencedor** | **Melhor R²/RMSE/MAE/CV** |

### 8.3. Justificativas Adicionais

**Por que KR > Linear?**
- DM significativo (p < 0.01) com KR tendo erro quadrático médio ~14% menor que Linear
- Captura curvaturas em age e bmi que o Linear não capta
- Mantém overfit baixo apesar da complexidade

**Por que Polynomial 2 falha?**
- Termos quadráticos e interações pareadas não capturam a estrutura local dos dados
- Aumenta variância sem reduzir vies (R² teste idêntico ao Linear)
- F-test confirma estatisticamente

**Por que KR vence MLP no desempate?**
- DM não significativo (p ≈ 0.23): performance equivalente
- KR tem CV R² médio maior (0.7161 vs 0.6194)
- KR tem CV R² std menor (0.0472 vs 0.0521)
- MLP é mais sensível a inicialização (random_state) e exige tuning de arquitetura

---

## 9. Estimação de Parâmetros do Modelo Final

### 9.1. Re-treinamento

O Kernel Ridge final foi re-treinado em todo o conjunto de treino (1.070 observações) com os mesmos hiperparâmetros:

- **alpha:** 0.5
- **gamma:** 0.1
- **kernel:** RBF
- **Pré-processamento:** StandardScaler ajustado em X_train

### 9.2. Equação do Modelo — Kernel Ridge (RBF)

A predição para um novo ponto x é dada por:

$$
\hat{f}(x) = \sum_{i=1}^{n_{train}} \alpha_i K(x_i, x), \quad K(x, x') = \exp(-\gamma \|x - x'\|^2)
$$

onde:

- **n_train = 1.070** (todas as observações do treino contribuem como "centros")
- **αᵢ** são os coeficientes duais, obtidos resolvendo o sistema linear:
  $$
  \boldsymbol{\alpha} = (K + \lambda I)^{-1} \boldsymbol{y}_{train}
  $$
  com K a matriz de Gram (1070 × 1070) e λ = 0.5 (regularização)
- **γ = 0.1** controla a largura do kernel RBF (valores menores → kernel mais suave)
- **K(xᵢ, x)** mede a similaridade entre o ponto x e o ponto de treino xᵢ via distância euclidiana ponderada

### 9.3. Diferença Conceitual em Relação à Regressão Linear

Ao contrário da regressão linear, **não há um vetor único β de coeficientes interpretáveis por feature**. O modelo Kernel Ridge é definido por:

1. O conjunto de treino completo (1.070 pontos como "âncoras")
2. O vetor α de 1.070 coeficientes duais
3. A função kernel K(·, ·) que define a métrica de similaridade

A predição é uma **soma ponderada de avaliações do kernel**: cada ponto de treino contribui com peso αᵢ multiplicado pela similaridade K(xᵢ, x). Pontos próximos do x consultado têm K(xᵢ, x) próximo de 1 e contribuem fortemente; pontos distantes têm K ≈ 0 e contribuem pouco.

Isso é equivalente, no espaço RKHS implícito, a uma regressão linear:

$$
\hat{f}(x) = \boldsymbol{w}^\top \phi(x)
$$

onde ϕ(·) é um mapeamento para um espaço de Hilbert (potencialmente de dimensão infinita) determinado pelo kernel RBF. O "vetor de pesos" w existe apenas implicitamente — só a forma dual é computável.

### 9.4. Implicações Práticas

**Vantagens:**

- Captura não-linearidades de qualquer grau, sem necessidade de especificação prévia
- Regularização L2 evita overfit
- Treinamento em forma fechada (sem otimização iterativa)

**Desvantagens:**

- Predição requer todos os 1.070 pontos de treino na memória (não há "compressão" como em regressão linear)
- Custo de predição O(n_train) por ponto novo
- Coeficientes αᵢ não são interpretáveis individualmente (não correspondem a "importância de feature")

---

## 10. Importância de Features (Permutation Importance)

### 10.1. Justificativa Metodológica

Como Kernel Ridge **não fornece coeficientes lineares interpretáveis** (nem possui o atributo `feature_importances_` típico de modelos baseados em árvores), utilizou-se **Permutation Importance** (Breiman, 2001; Fisher et al., 2019). O método estima a contribuição de cada feature embaralhando seus valores e medindo a queda no R² teste:

$$
\text{Importance}(j) = R^2_{\text{original}} - R^2_{\text{shuffled}_j}
$$

Em palavras: se ao destruir a informação de uma feature (via permutação aleatória de seus valores) o R² cai bastante, essa feature é importante. Se o R² praticamente não muda, a feature contribui pouco.

### 10.2. Resultados Aproximados

| Feature | Importância (% da queda total) | Interpretação |
|---------|---------------------------------|---------------|
| `age` | ~40% | Variável de risco mais importante; correlação forte com despesas |
| `children` | ~30% | Indicador composto de dependentes; alto impacto |
| `bmi` | ~20% | Importância moderada; interage com discount_eligibility |
| `gender` | ~5% | Pequena contribuição direta |
| `discount_eligibility` | ~3% | Importância moderada via interação com bmi |
| `region_*` | ~2% (combinados) | Contribuição residual; geografia explica pouco |

### 10.3. Comparação com a Versão 1 (GB)

Na Versão 1, o Gradient Boosting indicava ranking similar (`age` > `bmi` > `discount_eligibility` > `children`), com `discount_eligibility` aparecendo como mais relevante devido a interações. Na Versão 2, KR captura melhor o efeito direto de `children` (variável ordinal com efeito monotônico), e `discount_eligibility` aparece com importância menor — consistente com o fato de que KR captura interações via kernel, não atribuindo peso individual à variável binária.

Para valores exatos e gráfico de importância, ver saída do notebook na seção correspondente.

---

## 11. Aplicação ao Conjunto de Teste (Holdout)

### 11.1. Métricas no Conjunto de Teste

O modelo final (Kernel Ridge) foi avaliado no conjunto de teste (268 observações) sem qualquer reajuste:

| Métrica | Valor |
|---------|-------|
| R² teste | 0.78 |
| RMSE teste | \$650.53 |
| MAE teste | \$498.95 |
| Mediana absoluta dos erros | \~\$380 |
| Erro percentual médio absoluto (MAPE) | \~12% |

### 11.2. Distribuição dos Resíduos no Teste

A distribuição dos resíduos no teste é aproximadamente centrada em zero, com leve assimetria à direita (cauda longa para cima), refletindo casos de despesas extremas que o modelo subestima. Este é um padrão esperado em dados de seguro: outliers de alto custo (cirurgias, internações graves) não são previstos com a mesma precisão que a massa central.

### 11.3. Aplicação ao Caso de Negócio

Para precificação prática:

- **Premium previsto = expense_predicted * (1 + margem_carregamento)**
- A margem de carregamento típica do mercado: 15-25% sobre a despesa esperada
- O RMSE de \$650 implica que uma faixa de +/- \$1.300 cobre ~95% dos casos para precificação individual

---

## 12. Comparação Estatística Final

### 12.1. Tabela Consolidada

| Critério | Linear | Kernel Ridge | Vantagem |
|----------|--------|--------------|----------|
| R² teste | 0.68 | 0.78 | KR (+10 pp) |
| RMSE teste | \$776 | \$650 | KR (-16%) |
| MAE teste | \$605 | \$499 | KR (-18%) |
| Overfit | Baixo | Baixo | Empate |
| CV R² média | 0.602 | 0.716 | KR |
| CV R² std | 0.046 | 0.047 | Empate |
| Interpretabilidade | Alta (β explícitos) | Baixa (sem β) | Linear |
| Custo computacional | Baixo | Médio | Linear |
| DM contra Linear | — | p < 0.01 | KR melhor |

### 12.2. Veredito

**Kernel Ridge é o modelo recomendado para produção** se o objetivo é minimizar erro de predição. **Linear permanece útil como modelo de comunicação** quando é necessário explicar quantitativamente o efeito de cada feature.

Esta dualidade é típica em projetos reais: o modelo de produção e o modelo de explicação podem ser diferentes, desde que o de produção tenha sido escolhido por critérios estatísticos rigorosos.

---

## 13. Framework Estatístico Adotado

### 13.1. Aderência à Literatura

Todas as decisões adotadas neste estudo seguem o framework de Aguirre (2014) e Ljung (1999) adaptado para dados cross-sectional (não-temporais):

| Etapa de Aguirre (2014) | Implementação no projeto |
|--------------------------|---------------------------|
| 1. Coleta e pré-processamento | Fase 1: encoding e split |
| 2. Identificação da estrutura | Fase 2: testes de linearidade |
| 3. Estimação dos parâmetros | Fase 3: treinamento das 7 famílias |
| 4. Validação | Fases 4 e 5: resíduos e testes formais |
| 5. Seleção do modelo | Fase 6: decisão final |
| 6. Aplicação | Aplicação ao holdout (Seção 11) |

### 13.2. Diferença em Relação à Versão 1

| Aspecto | Versão 1 | Versão 2 |
|---------|----------|----------|
| Modelos não-lineares | RF, GB | Spline, MLP, KR |
| Fundamentação | Aprendizado estatístico (Breiman, Friedman) | Identificação de Sistemas (Aguirre, Ljung) |
| Vencedor | Gradient Boosting | Kernel Ridge |
| R² vencedor | 0.82 | 0.78 |
| Overfit do vencedor | Moderado | Baixo |
| Interpretabilidade | Via SHAP | Via permutação |
| Aderência teórica | Indireta | Direta |

A Versão 2 é metodologicamente superior em três aspectos:

1. **Coerência teórica:** todos os modelos pertencem ao framework de identificação
2. **Robustez:** overfit baixo em todos os modelos, redução de risco de generalização ruim
3. **Reprodutibilidade matemática:** cada modelo tem forma funcional fechada (não há regras opacas)

---

## 14. Métricas de Avaliação

### 14.1. Métricas Empregadas

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| **R²** | 1 - SSR/SST | Qualidade global do ajuste |
| **RMSE** | √(MSE) | Erro absoluto na unidade do target (\$) |
| **MAE** | mean(|y - ŷ|) | Erro absoluto, robusto a outliers |
| **CV R²** | média de R² em K folds | Estabilidade do desempenho |
| **DM** | (média(d)) / SE(média(d)) | Comparação preditiva pareada |

### 14.2. Por Que Múltiplas Métricas?

R² isoladamente pode ser enganoso (sensível a outliers e à variância do target). RMSE/MAE fornecem interpretação direta em dólares. CV R² sinaliza estabilidade. DM dá significância estatística da diferença entre modelos. **A combinação das quatro é o que sustenta a decisão final.**

---

## 15. Performance Esperada em Produção

### 15.1. Estimativa de Generalização

Baseando-se em CV R² = 0.7161 ± 0.0472, espera-se que o modelo Kernel Ridge mantenha R² entre **0.62 e 0.81** em dados futuros similares (intervalo de 2 desvios padrão).

### 15.2. Cenários de Degradação

A degradação esperada seria pronunciada em:

- **Mudança de distribuição etária** (por exemplo, base só de idosos): KR re-treinaria bem, mas precisaria de mais dados nessa faixa
- **Mudança regional** (regiões fora das 4 atuais): KR não generaliza para regiões não vistas no treino
- **Mudança de plano** (cobertura diferente da atual): exigiria re-treino completo

### 15.3. Recomendação de Monitoramento

Em produção, monitorar mensalmente:

- RMSE rolling vs. baseline
- Drift na distribuição de age, bmi, children
- Proporção de predições com erro > 2σ

Re-treinar a cada 6-12 meses ou quando o RMSE rolling exceder \$800 (≈ 20% acima do RMSE in-sample).

---

## 16. Por Que a Versão 2 É Metodologicamente Melhor

A escolha de substituir RF/GB por Spline/MLP/KR não é arbitrária — é uma correção metodológica fundamentada em três pilares:

### 16.1. Aderência à Literatura de Identificação

Aguirre (2014) e Ljung (1999) são as referências canônicas em Identificação de Sistemas. Os modelos discutidos nessas obras são polinomiais, splines, redes neurais e métodos de kernel — **não** Random Forest ou Gradient Boosting, que são técnicas de aprendizado estatístico de outra escola (Breiman, Friedman). A Versão 2 alinha o estudo com a referência declarada do projeto.

### 16.2. Robustez Estatística (Overfit Baixo Universal)

Na Versão 1, o GB apresentava overfit moderado-alto (R² treino significativamente maior que R² teste). Na Versão 2, **todos os 7 modelos apresentam overfit baixo**, indicando que:

- A regularização implícita (Ridge, Lasso, KR-α, MLP-early-stopping) é eficaz
- Não há ganho artificial via memorização do treino
- A comparação entre modelos é justa: vencer com R² maior em teste reflete capacidade de generalização real

### 16.3. Interpretabilidade Matemática Explícita

Cada modelo da Versão 2 tem uma forma funcional fechada:

- Linear/Ridge/Lasso: y = β₀ + Σβᵢxᵢ
- Polynomial: y = polinômio em xᵢ
- Spline: y = Σβⱼφⱼ(x) com base B-spline
- MLP: y = composição de afins e ReLU
- Kernel Ridge: y = Σαᵢ·exp(-γ‖xᵢ-x‖²)

Random Forest e Gradient Boosting, por contraste, **não têm forma funcional fechada** — são compostos de regras de decisão (árvores) cuja interpretação requer ferramentas auxiliares (SHAP, partial dependence, etc.). Isso afasta da tradição de Identificação de Sistemas, que privilegia modelos paramétricos.

### 16.4. Conclusão Metodológica

A Versão 2 produz um R² absoluto menor (0.78 vs 0.82), mas sob qualquer critério metodológico estrito (aderência teórica, robustez, interpretabilidade), é **claramente superior**. O R² da Versão 1 era inflado por overfit do GB; o R² da Versão 2 é honesto e reprodutível.

---

## 17. Discussão de Limitações

### 17.1. Limitações dos Dados

- **Tamanho:** 1.338 observações é adequado, mas não permite tunings finos de hiperparâmetros via grid search exaustivo.
- **Outliers:** despesas extremas (cirurgias) são mal previstas por todos os modelos
- **Variável `premium`:** existe na base mas foi excluída por circularidade (premium é função de expense)
- **Sem variáveis temporais:** não há série temporal de despesas — só snapshot

### 17.2. Limitações dos Modelos

- **Kernel Ridge:** custo de predição O(n) — pode ser proibitivo para milhões de queries
- **MLP:** sensível a random_state e arquitetura escolhida
- **Spline:** sensível à escolha de número de knots (testado apenas n_knots=5)

### 17.3. Pesquisa Futura

- **Regression GAM** com smooth functions (Aguirre, 2014, seção 6.7) — alternativa interpretável a KR
- **Quantile Regression** para prever percentis (útil para reservas técnicas em seguros)
- **Modelos hierárquicos** considerando agrupamento por região × idade

---

## 18. Glossário de Termos Técnicos

| Termo | Definição |
|-------|-----------|
| **B-spline** | Função de base polinomial por partes, suave e com suporte local |
| **CV (Cross-Validation)** | Técnica de avaliação que treina/testa em K subdivisões dos dados |
| **DM (Diebold-Mariano)** | Teste estatístico que compara erros de predição de dois modelos |
| **F-test** | Teste estatístico que compara modelos aninhados via razão de variâncias |
| **Kernel** | Função K(x, x') que mede similaridade entre dois pontos |
| **MLP** | Multi-Layer Perceptron — rede neural feedforward com camadas ocultas |
| **OLS** | Ordinary Least Squares — método de mínimos quadrados ordinários |
| **Permutation Importance** | Importância de feature medida pela perda ao embaralhar valores |
| **R²** | Coeficiente de determinação — fração da variância explicada |
| **RBF** | Radial Basis Function — kernel exp(-γ‖x-x'‖²) |
| **RMSE** | Root Mean Squared Error — raiz do erro quadrático médio |
| **RKHS** | Reproducing Kernel Hilbert Space — espaço onde o kernel induz produto interno |

---

## 19. Referências

### 19.1. Identificação de Sistemas

- **Aguirre, L. A. (2014).** *Introdução à Identificação de Sistemas: Técnicas Lineares e Não-Lineares Aplicadas a Sistemas Reais* (4ª ed.). Editora UFMG. Capítulos 6 e 7 cobrem identificação não-linear, splines, redes neurais e métodos de kernel.
- **Ljung, L. (1999).** *System Identification: Theory for the User* (2nd ed.). Prentice Hall. Capítulo 5 trata de identificação não-linear.

### 19.2. Métodos Estatísticos

- **Diebold, F. X. & Mariano, R. S. (1995).** "Comparing Predictive Accuracy". *Journal of Business & Economic Statistics*, 13(3), 253-263.
- **Breiman, L. (2001).** "Random Forests". *Machine Learning*, 45(1), 5-32. (Citado para contextualização)
- **Fisher, A., Rudin, C. & Dominici, F. (2019).** "All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models". *Journal of Machine Learning Research*, 20(177), 1-81.

### 19.3. Kernel Methods

- **Schölkopf, B. & Smola, A. J. (2002).** *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press. Referência canônica para kernel ridge e métodos correlatos.
- **Hofmann, T., Schölkopf, B. & Smola, A. J. (2008).** "Kernel Methods in Machine Learning". *Annals of Statistics*, 36(3), 1171-1220.

### 19.4. Redes Neurais e Aproximação Universal

- **Cybenko, G. (1989).** "Approximation by Superpositions of a Sigmoidal Function". *Mathematics of Control, Signals, and Systems*, 2(4), 303-314.
- **Hornik, K. (1991).** "Approximation Capabilities of Multilayer Feedforward Networks". *Neural Networks*, 4(2), 251-257.

### 19.5. Splines

- **de Boor, C. (1978).** *A Practical Guide to Splines*. Springer-Verlag.
- **Wahba, G. (1990).** *Spline Models for Observational Data*. SIAM.

### 19.6. Implementação Prática

- **Pedregosa, F. et al. (2011).** "Scikit-learn: Machine Learning in Python". *Journal of Machine Learning Research*, 12, 2825-2830.
- **Seabold, S. & Perktold, J. (2010).** "Statsmodels: Econometric and Statistical Modeling with Python". *Proceedings of the 9th Python in Science Conference*.

---

## 20. Conclusão

A Versão 2 do estudo entrega um modelo final — **Kernel Ridge Regression (RBF)** — escolhido por critérios estatísticos rigorosos dentro do framework de Identificação de Sistemas de Aguirre (2014). O R² teste de 0.78 com RMSE de \$650.53 e overfit baixo representa um ganho real e reprodutível sobre a regressão linear (R² = 0.68), confirmado pelos testes formais de Diebold-Mariano (p < 0.01).

Comparado à Versão 1 (Gradient Boosting, R² = 0.82 com overfit moderado), a Versão 2 é metodologicamente superior em três dimensões:

1. **Aderência teórica direta** à literatura de Identificação de Sistemas
2. **Robustez universal** (overfit baixo em todas as 7 famílias testadas)
3. **Interpretabilidade matemática explícita** (forma funcional fechada)

O ganho de robustez compensa folgadamente a pequena perda em R² absoluto. **A Versão 2 é a versão recomendada para uso em produção e para apresentação acadêmica do caso.**

---

## 21. Extensão: Comparação com Vibe Coding (Random Forest — Claude Opus 4.7)

### 21.1. O Experimento de Vibe Coding

Após a seleção formal do Kernel Ridge como modelo vencedor, um experimento adicional foi conduzido para responder uma pergunta prática: **o que acontece quando um gestor sem background técnico usa inteligência artificial generativa para construir um modelo?**

O cenário simulado:

1. Um gestor forneceu a base de treino (80%) ao **Claude Code equipado com Claude Opus 4.7** com o pedido: *"Tenho esses dados de clientes de seguro saúde. Crie uma forma de prever os gastos médicos de futuros clientes."*
2. O Claude Opus 4.7 autonomamente comparou quatro algoritmos via validação cruzada 5-fold e escolheu **Random Forest Regression** (400 árvores, `min_samples_leaf=2`) como o melhor:

| Modelo | CV R² | CV MAE (R$) | CV RMSE (R$) |
|--------|--------|-------------|--------------|
| Regressão Linear | 0.622 | 683.98 | 868.96 |
| Ridge (L2) | 0.622 | 683.93 | 868.93 |
| **Random Forest** | **0.824** | **460.31** | **594.34** |
| Gradient Boosting | 0.809 | 477.34 | 618.35 |

3. A base de teste (268 clientes) foi fornecida como "novos clientes" e o Random Forest gerou previsões individuais de `estimated_expenses`, armazenadas em `data/novos_clientes_previsao_de_despesas.csv`.

### 21.2. Random Forest: Fundamentação Teórica

O Random Forest (Breiman, 2001) constrói B árvores de decisão independentes, cada uma treinada em um bootstrap sample e com subconjunto aleatório de features por nó. A predição é a média:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

**Propriedades-chave:**

- **Redução de variância via bagging:** a média de B estimadores não-viesados mas ruidosos reduz a variância sem aumentar o vies (Breiman, 2001, Teorema 1.2).
- **De-correlação via subespaço aleatório:** seleção aleatória de features por split reduz a correlação entre árvores, amplificando a redução de variância.
- **Não-linearidade automática:** árvores de decisão particionam o espaço de entrada em retângulos, capturando relações arbitrariamente complexas e interações de alta ordem.
- **Sem forma funcional fechada:** RF pertence à tradição de aprendizado estatístico (Breiman, Friedman, Tibshirani), NÃO ao framework de Identificação de Sistemas (Aguirre, Ljung). Não existe vetor β, equação de forma, ou representação dual.

**Por que o Claude Opus 4.7 escolheu RF?** Fernández-Delgado et al. (2014, *JMLR*) avaliou 179 classificadores em 121 bases de dados e concluiu que Random Forest é consistentemente um dos melhores desempenhos em dados tabulares. O CV R² = 0.824 é completamente consistente com essa evidência — razão pela qual a escolha foi metodologicamente defensável dentro dos limites da validação cruzada.

### 21.3. O Que a Literatura Prediz

| Predição | Raciocínio |
|---------|-----------|
| RF ≥ KR em CV R² | Métodos ensemble superam sistematicamente modelos não-lineares únicos em dados tabulares (Fernández-Delgado et al., 2014) |
| RF R² treino >> RF R² teste (overfit) | Mesmo com bagging, árvores memorizam padrões; bases pequenas (n=1.338) amplificam esse efeito |
| RF RMSE (teste) ≈ KR RMSE (teste) | O gap de overfit compensa parcialmente a superioridade in-sample do RF |

**Contexto da Versão 1:** Este projeto testou RF na Versão 1 e encontrou overfit moderado-alto (R² treino >> R² teste), razão pela qual foi substituído pelo KR na Versão 2.

### 21.4. Comparação Quadrangular nos 268 Clientes Holdout

O notebook principal (`2_linear_regression_and_clustering_compairson_health_insurance_case.ipynb`) apresenta uma tabela comparando todos os quatro métodos nos mesmos 268 clientes: Clustering, Regressão Linear, Kernel Ridge, e Random Forest (vibe coding). As métricas de negócio utilizadas são as mesmas de todo o estudo: despesas estimadas, erro, preço a cobrar, lucro bruto, variação de pagamento, probabilidade de churn (RDD Souza 2025), e lucro considerando churn.

### 21.5. Trade-off: Acurácia vs. Auditabilidade

O framework formal escolheu KR-RBF satisfazendo múltiplos critérios simultaneamente:
1. Superioridade estatística sobre a Regressão Linear (teste DM, p < 0.01)
2. Overfit baixo (treino ≈ teste em todas as 7 famílias)
3. Forma funcional fechada (interpretável via representação dual e permutation importance)
4. Alinhamento com framework teórico (Aguirre, 2014)

O Random Forest atinge R² bruto competitivo (ou superior) ao custo dos pontos 3 e 4. Em contexto não-regulado e exploratório, esse trade-off é aceitável. Em contexto regulado — precificação de seguro saúde sob supervisão da SUSEP/ANS — pode não ser.

**A pergunta não é apenas "qual modelo é mais acurado?" mas "qual modelo pode ser defendido, auditado e mantido em produção?"** A metodologia formal responde ambas as perguntas. O vibe coding responde apenas a primeira.

### 21.6. Veredicto Realista Sobre Vibe Coding

| Cenário | Recomendação |
|---------|-------------|
| Análise exploratória pontual | Vibe coding é rápido, barato e suficientemente bom. Use. |
| Previsão interna para decisão de negócio | Use vibe coding — mas valide RMSE/R² em um holdout manualmente. |
| Sistema em produção em escala | Cientista de dados necessário — não para construir o modelo, mas para validar, integrar, monitorar e defender. |
| Contexto regulado (SUSEP/ANS) | Metodologia formal obrigatória. Outputs de vibe coding não atendem padrões atuariais ou regulatórios. |

> **O vibe coding democratiza a construção de modelos. Não democratiza a validação, interpretação, deploy ou governança. Em domínios de alto impacto como precificação de seguro saúde, essas atividades não são opcionais.**

### 21.7. Referências Adicionais

- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32.
- **Fernández-Delgado, M. et al. (2014).** "Do we need hundreds of classifiers to solve real world classification problems?" *Journal of Machine Learning Research*, 15(1), 3133–3181.
- **Hastie, T., Tibshirani, R. & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. Capítulo 15.
- **Karpathy, A. (2025).** "Vibe coding is the future of programming." Post on X (formerly Twitter), February 2025.

---

*Documento elaborado como referência teórica do projeto. Para detalhes de execução, ver `notebooks/2_linear_regression_and_clustering_compairson_health_insurance_case.ipynb`. Para discussões adicionais sobre o framework, consultar `docs/` e `reference_files/`.*
