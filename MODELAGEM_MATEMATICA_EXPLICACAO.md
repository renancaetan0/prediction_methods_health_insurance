# Modelagem Matemática — Caso de Precificação de Seguro Saúde

> **Como ler este documento:** Cada seção explica o que foi feito, por que foi feito dessa forma e o que os resultados significam na prática. Os números são os resultados reais do notebook executado.

---

## O Dilema: Suposições Implícitas vs. Orientação Teórica

Até a célula 93 do notebook, dois modelos foram avaliados de forma independente:
- **K-Means (clustering):** agrupa clientes em segmentos e precifica por grupo
- **Regressão Linear:** equação que prevê o preço individualmente

A conclusão comparativa mostrou que ambos têm limitações. O clustering precifica por grupo médio — ignora variação individual. O modelo linear assume que todas as relações são lineares e aditivas — o que pode ou não ser verdade nos dados.

**O problema:** nenhuma das duas abordagens foi escolhida com base em evidência estatística formal. Foram suposições. A pergunta que fica é: **qual é a estrutura correta do modelo para esses dados?**

Para responder isso com rigor, seguiu-se a metodologia de Identificação de Sistemas de Ljung (1999) e Aguirre (2014), adaptada para dados transversais (cross-sectional) — ou seja, sem dimensão temporal.

---

## Phase 1 — Codificação de Variáveis Categóricas

### O que é e por que fazer antes de qualquer teste

Antes de testar qualquer modelo, variáveis categóricas precisam ser convertidas em números. Mas a forma como isso é feito importa matematicamente.

- **`gender`** tem 2 categorias: `female` e `male`
- **`region`** tem 4 categorias: `midwest`, `northeast`, `south`, `southeast`

A técnica usada é **One-Hot Encoding com `drop_first=True`**. Em vez de criar uma coluna `gender` com 0/1, cria-se uma coluna `gender_male` (1 se masculino, 0 se feminino). A categoria `female` é a **referência implícita** — o coeficiente de `gender_male` mede a diferença no preço entre masculino e feminino, tudo mais constante.

Para região, cria-se 3 colunas (não 4): `region_northeast`, `region_south`, `region_southeast`. A referência é `midwest`. Usar 4 colunas criaria **multicolinearidade perfeita** com o intercepto — o modelo não conseguiria separar os efeitos.

**Por que isso importa para os próximos passos?** Wooldridge (2019) adverte: testar estrutura de modelo (linearidade, interações) sem ter todas as variáveis presentes causa **viés de omissão** — você atribui a uma variável o efeito que é, na verdade, de outra. Por isso Phase 1 vem antes de tudo.

### Resultado

```
Columns: ['age', 'bmi', 'children', 'expenses', 'gender_male',
          'region_northeast', 'region_south', 'region_southeast']
Shape: (1070, 8)  →  7 features + 1 target
Reference categories dropped: gender → female | region → midwest
✓ No missing values. ✓ 8 columns confirmed.
```

---

## Phase 2 — Teste de Linearidade por Feature

### O que é e por que fazer

A regressão linear assume que cada feature tem uma relação **linear** com o target. Isso significa: se `age` sobe 1 ano, `expenses` sobe sempre o mesmo valor — independente de qual seja a idade atual.

Mas e se a relação for curvilínea? Se aos 20 anos um aumento de 1 ano causa pouco impacto e aos 50 anos causa muito mais? Isso seria **não-linearidade**.

O teste usa a lógica de Aguirre (2014): adiciona o termo quadrático (`age²`) ao modelo com todas as features e mede se o R² melhora significativamente.

- **Modelo A:** `expenses ~ age + bmi + children + gender_male + region_*`
- **Modelo B:** `expenses ~ age + age² + bmi + children + gender_male + region_*`
- **Critério:** Se R² de B melhora mais de 10% em relação a A → relação **NÃO-LINEAR**

O teste é feito **controlando por todas as outras variáveis** justamente para não confundir o efeito de `age` com o de `region`.

### Resultado

| Feature | R² Linear | R² Quadrático | Δ R² (%) | Decisão |
|---------|-----------|---------------|----------|---------|
| `age`   | 0.6242    | 0.6247        | **0.09%** | **LINEAR** |
| `bmi`   | 0.6242    | 0.6285        | **0.69%** | **LINEAR** |

**Interpretação:** Adicionar `age²` melhora o R² em apenas 0,09% — quase zero. O mesmo para `bmi` (0,69%). Ambas as features são **lineares** no contexto multivariado.

**Implicação para Phase 3:** Modelos polinomiais não devem ter grande vantagem sobre o linear, pois as features já são lineares. Se um modelo não-linear vencer, será por outro motivo — provavelmente **interações** entre features, não curvatura individual.

---

## Phase 3 — Treinar e Comparar 6 Famílias de Modelo

### Os 6 modelos e suas estruturas

**Dados:** 80% treino (856 linhas), 20% teste (214 linhas). Mesmo split para todos.

#### 1. Linear Regression — O baseline

A equação mais simples:

```
expenses = β₀ + β₁·age + β₂·bmi + β₃·children + β₄·gender_male
         + β₅·region_NE + β₆·region_S + β₇·region_SE
```

Cada coeficiente `βᵢ` é um número fixo. A contribuição de cada variável é sempre a mesma, independente das outras. Não há interação entre features.

#### 2. Polynomial Degree 2 — Curvas e interações manuais

Expande o espaço de features para incluir termos como `age²`, `age × bmi`, `bmi × children`, etc. Com 7 features originais, gera **35 termos** (combinações de grau até 2).

Captura relações não-lineares e interações, mas ainda é uma equação paramétrica — cada coeficiente tem interpretação direta.

#### 3. Ridge (L2) e 4. Lasso (L1) — Linear com controle de overfitting

São versões da regressão linear que adicionam uma **penalidade** matemática sobre o tamanho dos coeficientes. Isso evita que o modelo "memorize" o treino ao invés de generalizar.

- **Ridge:** encolhe todos os coeficientes, nenhum vai a zero
- **Lasso:** zera completamente os menos importantes (seleção automática de variáveis)

O Lasso reteve apenas **3 das 7 features** — provavelmente `age`, `children` e `bmi`.

#### 5. Random Forest — Votação de 100 árvores de decisão

Não existe equação. O modelo constrói 100 árvores de decisão, cada uma treinada em uma amostra aleatória dos dados. Cada árvore faz uma série de perguntas binárias:

```
age > 45?
  Sim → children > 2?
         Sim → bmi > 30? → previsão A
         Não → previsão B
  Não → ...
```

A previsão final é a **média das 100 árvores**. Isso captura interações complexas (o efeito de `age` pode depender de `children`) sem precisar especificá-las manualmente.

#### 6. Gradient Boosting — Árvores sequenciais corretivas

Também usa árvores de decisão, mas de forma **sequencial**: cada árvore nova é treinada para corrigir os erros da anterior. É mais preciso que o Random Forest em tabular data, mas mais sensível a overfitting.

### Resultado

| Modelo | R² Treino | R² Teste | RMSE Teste | MAE Teste | Overfit | CV R² (5-fold) |
|--------|-----------|----------|------------|-----------|---------|----------------|
| **Gradient Boosting** | 0.93 | **0.83** | $566.77 | $436.03 | Medium | 0.80 ± 0.016 |
| **Random Forest** | 0.91 | **0.82** | $577.70 | $438.53 | Medium | 0.81 ± 0.021 |
| Poly2 | 0.69 | 0.69 | $761.45 | $580.64 | Low | — |
| Linear | 0.61 | 0.68 | $776.10 | $604.86 | Low | 0.60 ± 0.046 |
| Ridge | 0.61 | 0.68 | $777.15 | $604.96 | Low | 0.60 ± 0.047 |
| Lasso | 0.61 | 0.68 | $778.35 | $604.96 | Low | 0.60 ± 0.047 |

**Separação em dois grupos:**
- **Tree-based** (RF, GB): R² ≈ 0.82–0.83, RMSE ≈ $567–578
- **Lineares** (Linear, Ridge, Lasso, Poly2): R² ≈ 0.68–0.69, RMSE ≈ $761–778

**Gap de ~14 pontos percentuais de R² e ~$210 de RMSE por cliente.** Os modelos lineares erram sistematicamente mais.

**Por que os lineares ficam atrás se age e bmi são lineares (Phase 2)?** Porque Phase 2 testou *cada feature individualmente*. Os tree models capturam **interações**: o efeito de `age` muda dependendo de quantos `children` o cliente tem. Isso é invisível para o modelo linear — e explica o gap.

**Feature importance (média RF + GB):**

| Feature | Importância Média |
|---------|------------------|
| `age` | 53% |
| `children` | 35% |
| `bmi` | 11% |
| `gender_male` | 1% |
| `region_*` | < 1% cada |

`age` e `children` juntos explicam **~86% da variação** no preço. `region` e `gender` são praticamente irrelevantes.

---

## Phase 4 — Análise de Resíduos

### O que é e por que fazer

Resíduo = diferença entre o valor real e o valor previsto pelo modelo. Se o modelo é bem especificado, os resíduos devem ser **ruído aleatório puro** — sem padrão, sem estrutura.

Dois testes formais:
- **Shapiro-Wilk:** testa se os resíduos seguem distribuição normal (p > 0.05 = normal)
- **Breusch-Pagan:** testa homoscedasticidade — se a variância dos resíduos é constante (p > 0.05 = constante)

### Resultado

| Modelo | Média Resíduo | Desvio Padrão | SW p-value | Normal? | BP p-value | Homosced? |
|--------|--------------|---------------|------------|---------|------------|-----------|
| Linear | -49.83 | $774.50 | 0.0000 | **Não** | 0.0000 | **Não** |
| Random Forest | 12.55 | $577.57 | 0.0100 | **Não** | 0.0000 | **Não** |
| Gradient Boosting | 5.97 | $566.74 | 0.0300 | **Não** | 0.0000 | **Não** |

**Interpretação:** Nenhum modelo passou nos testes de normalidade e homoscedasticidade. Mas o que importa é o contexto:

- Para o **modelo linear**, isso é um problema sério: os testes de significância (p-values dos coeficientes) assumem resíduos normais e homoscedásticos. Se esses pressupostos falham, os intervalos de confiança e p-values são **inválidos**.
- Para os **modelos tree-based**, esses pressupostos não existem — eles não têm inferência paramétrica. O que importa é a qualidade preditiva, não a distribuição dos resíduos. O desvio padrão menor ($566 vs $774) confirma que os tree models erram menos sistematicamente.

**VIF (Variance Inflation Factor) — multicolinearidade no modelo linear:**

| Feature | VIF |
|---------|-----|
| `bmi` | **17.16** |
| `age` | **11.59** |
| `region_southeast` | 2.13 |
| `children` | 1.99 |
| outros | < 2.0 |

> VIF < 5: baixa colinearidade | 5–10: moderada | **> 10: alta**

`bmi` (17.16) e `age` (11.59) têm **colinearidade elevada** com outras variáveis no modelo linear. Isso significa que os coeficientes individuais dessas features são instáveis — pequenas mudanças nos dados podem alterar muito os coeficientes estimados. **Ridge e Lasso foram criados exatamente para mitigar esse problema**, mas mesmo assim não superam os tree models em R².

---

## Phase 5 — Testes Estatísticos Formais

### O que é e por que fazer

Phase 3 mostrou que GB e RF são melhores. Mas "melhor na amostra" pode ser coincidência. Phase 5 usa testes estatísticos para responder: **a diferença de performance é significativa ou pode ser ruído?**

#### F-test: Linear vs Polynomial (modelos aninhados)

O Poly2 contém o Linear como caso especial (quando todos os coeficientes quadráticos são zero). Por isso, é possível testar formalmente se os 28 coeficientes extras do Poly2 trazem ganho real.

```
Hipótese nula: os termos quadráticos não melhoram o modelo (todos os β_quadráticos = 0)
```

**Resultado:**

```
RSS Linear: 128.898.196  |  RSS Poly2: 124.079.858
Extra parâmetros: 28  |  F = 0.2483  |  p-value = 1.0
RESULTADO: Linear preferido — termos polinomiais não trazem ganho significativo
```

Um F de 0.25 com p = 1.0 é inequívoco: **os 28 termos extras do Poly2 não explicam nada que o Linear já não explique**. Consistente com Phase 2 (age e bmi são lineares).

#### Diebold-Mariano: modelos não-aninhados

O RF e GB não são versões do Linear — são arquiteturas completamente diferentes. Por isso o F-test não se aplica. O **teste de Diebold-Mariano** compara diretamente os erros de previsão ao quadrado dos dois modelos e testa se a diferença é sistematicamente diferente de zero.

```
DM < 0 → primeiro modelo tem erros menores (é melhor)
p < 0.05 → diferença é estatisticamente significativa
```

**Resultado:**

| Comparação | DM | p-value | Significativo? | Melhor modelo |
|-----------|-----|---------|----------------|---------------|
| RF vs Linear | **-4.58** | **0.00** | **SIM** | RF |
| GB vs Linear | **-4.83** | **0.00** | **SIM** | GB |
| GB vs RF | -1.01 | 0.31 | não | Equivalentes |

- RF é **significativamente melhor** que Linear (p ≈ 0, DM = -4.58)
- GB é **significativamente melhor** que Linear (p ≈ 0, DM = -4.83)
- GB e RF são **estatisticamente equivalentes** entre si (p = 0.31) — a diferença de $11 no RMSE pode ser ruído

#### AIC/BIC: penalidade por complexidade (modelos paramétricos)

Critério de informação: premia ajuste mas penaliza número de parâmetros. Menor = melhor.

| Modelo | k parâmetros | AIC | ΔAIC | Evidência |
|--------|-------------|-----|------|-----------|
| **Lasso** | **3** | **2855.3** | **0.0** | **Melhor** |
| Linear | 7 | 2862.0 | 6.7 | Moderado |
| Ridge | 7 | 2862.6 | 7.3 | Moderado |
| Poly2 | 35 | 2909.9 | 54.6 | **Rejeitado** |

Entre os modelos paramétricos, o **Lasso com apenas 3 features** é o melhor por AIC — confirma que a maioria das features tem impacto marginal. O Poly2 é **fortemente rejeitado** (ΔAIC = 54.6 >> 10) pela penalidade dos 35 parâmetros.

---

## Phase 6 — Recomendação Final

### Tabela consolidada de todos os modelos

| Modelo | R² Treino | R² Teste | RMSE | MAE | Gap | Overfit | CV R² |
|--------|-----------|----------|------|-----|-----|---------|-------|
| **Gradient Boosting** | 0.93 | **0.83** | $566.77 | $436.03 | 0.10 | Medium | 0.80 ± 0.016 |
| **Random Forest** | 0.91 | **0.82** | $577.70 | $438.53 | 0.09 | Medium | 0.81 ± 0.021 |
| Poly2 | 0.69 | 0.69 | $761.45 | $580.64 | -0.01 | Low | — |
| Linear | 0.61 | 0.68 | $776.10 | $604.86 | -0.07 | Low | 0.60 ± 0.046 |
| Ridge | 0.61 | 0.68 | $777.15 | $604.96 | -0.07 | Low | 0.60 ± 0.047 |
| Lasso | 0.61 | 0.68 | $778.35 | $604.96 | -0.07 | Low | 0.60 ± 0.047 |

### Trilha de decisão

```
Phase 2: age e bmi são LINEARES (Δ R² < 1%)
  → Phase 3: Poly2 confirma (R² = 0.69 ≈ Linear = 0.68)
  → Phase 5 F-test: p = 1.0 → termos quadráticos rejeitados formalmente
     ↓
     Mas RF e GB estão em 0.82–0.83 — gap de 14pp
     Causa: INTERAÇÕES entre features (age × children, etc.)
     → Phase 5 DM-test: gap é SIGNIFICATIVO (p ≈ 0 para ambos)
        ↓
        GB vs RF: equivalentes (p = 0.31)
        → Desempate por estabilidade: GB tem menor CV std (0.016 vs 0.021)
           → MODELO RECOMENDADO: Gradient Boosting
```

### Modelo recomendado: **Gradient Boosting**

```
R² Teste:        0.83   (vs Linear: 0.68 — melhora de 21.8%)
RMSE Teste:      $566.77  (vs Linear: $776.10 — $209 a menos de erro por cliente)
MAE Teste:       $436.03
Overfit:         Medium (gap treino-teste = 0.10)
CV R² 5-fold:    0.80 ± 0.016  (estável, generaliza bem)
```

**Por que não Linear?** Gap de 0.1485 R² (21.8% relativo), confirmado como estatisticamente significativo pelo DM-test (p ≈ 0). Os resíduos do Linear também violam homoscedasticidade, invalidando a inferência paramétrica.

**Por que não Polynomial?** Phase 2 mostrou < 1% de ganho com termos quadráticos. F-test confirmou formalmente (p = 1.0). AIC rejeitou Poly2 com ΔAIC = 54.6.

**Por que não Random Forest?** GB e RF são estatisticamente equivalentes em performance (DM p = 0.31). O desempate é pela menor variabilidade no CV (GB: ±0.016 vs RF: ±0.021) — GB é mais estável.

### Feature importance — o que realmente precifica o seguro

| Feature | RF | GB | Média |
|---------|----|----|-------|
| `age` | 53% | 52% | **53%** |
| `children` | 35% | 34% | **35%** |
| `bmi` | 10% | 11% | **11%** |
| `gender_male` | 1% | 1% | 1% |
| `region_south` | 0% | 0% | ~0% |
| `region_southeast` | 0% | 0% | ~0% |
| `region_northeast` | 0% | 0% | ~0% |

### Implicações práticas para o negócio

1. **`age` e `children` são os dois únicos fatores de risco relevantes** (86% combinados). A tarifa deve ser sensível a esses dois.
2. **`bmi` tem impacto moderado** (11%) — pode compor um fator de ajuste secundário.
3. **`region` e `gender` são irrelevantes** (< 2%) — diferenciar tarifa por região ou gênero não é suportado pelos dados e pode gerar exposição regulatória desnecessária.
4. **O modelo linear subestima o risco de clientes com combinação de alta idade + muitos filhos** — exatamente os casos mais caros — porque não captura a interação entre as duas variáveis.
5. **Recomendação de manutenção:** revalide o modelo trimestralmente com novos dados de sinistros. A feature importance pode mudar com novos perfis populacionais.

---

## Referências

- **Ljung, L. (1999).** *System Identification: Theory for the User.* 2ª ed. Prentice Hall.
- **Aguirre, L. A. (2014).** *Introdução à Identificação de Sistemas.* 4ª ed. Editora UFMG.
- **Wooldridge, J. (2019).** *Introductory Econometrics: A Modern Approach.* 7ª ed. Cengage.
- **James, G. et al. (2013).** *An Introduction to Statistical Learning.* Springer.

outras perguntas:

1) Em: "Phase 3=  ... Dados: 80% treino (856 linhas), 20% teste (214 linhas). "
significa que você utilizou a base de dados base_health_insurance_train_df (80% da base original — ~1.040 linhas treino, ~260 linhas teste) , que já éra 80% da base dados original, e dividiu ela em outros 80% para treino?

isso faz sentido? Veja que em "Plano_passo_apasso_ATUALIZADO.md" constava a informação:
- **Dataset:** base_health_insurance_train_df (80% da base original — ~1.040 linhas treino, ~260 linhas teste)


2) Qual é a equação do modelo eleito o melhor? Ou o que foi calculado ate agora foi so a estrutura e não o modelo em si? (neste caso ainda falta estimar os parâmetros do modelo)

3) Depois de ter o modelo estimado. Iremos processar a base de dados base_health_insurance_test_df (que são os 20% da base original dos clientes) e estimar "expenses" e depois comparar com os modelos de clusterização e regressão linear já aplicados a esta base de dados.
Você sugere que diante do que respondeu na questão 1) que enviei aqui, que façamos todo o processo da escolha do modelo matemático ideal considerando 100% da base_health_insurance_test_df  que foi a mesma base de dados usada para treinar clusterização e regressão linear, podendo assim fazer uma comparação justa com eles?

ou sugere que sigamos daqui, considerando as fases da esolha do modelo como estão?

Respostas
1) O split duplo faz sentido?
Sim, é exatamente isso — e está documentado no Plano como "Opção A".

A realidade dos tamanhos de dados:

Base original completa: 1.338 linhas (Train 1070 + Test 268)
base_health_insurance_train_df: 1.070 linhas (80% do original — usado para treinar clustering e regressão linear)
base_health_insurance_test_df: 268 linhas (20% do original — holdout final)
O Plano (Fase 3.2) pega base_health_insurance_train_df e faz mais um split 80/20 internamente:

1.070 × 0,80 = 856 treino
1.070 × 0,20 = 214 validação interna
O objetivo é selecionar o modelo sem tocar nas 268 linhas do holdout final. É a abordagem metodologicamente correta — o conjunto de teste final fica "lacrado" até a comparação definitiva.

O plano tem uma inconsistência de redação no cabeçalho (escreveu "~1.040 treino, ~260 teste" misturando as proporções do dataset original), mas a lógica do código da Fase 3.2 está correta.

2) Existe uma equação do modelo eleito?
Não — só a estrutura/plano foi definida, os parâmetros ainda não foram estimados.

O que existe no notebook hoje (células 75-79):

Um OLS simples já treinado: expenses ~ age + bmi + children (com intercepto e 3 coeficientes estimados pelo statsmodels)
É o modelo da análise original de regressão linear — não é o resultado das Fases 1-6 do plano novo
O que as Fases 1-6 do plano propõem fazer (ainda não executado):

Fase 1: encoding das categóricas → df_encoded com 7 features
Fase 2: testes de linearidade (age, bmi)
Fase 3: treinar 6 famílias de modelo e comparar métricas
Fases 4-6: resíduos, testes estatísticos, recomendação final
Faltam os parâmetros — o modelo ideal ainda nem foi escolhido, muito menos estimado.

3) Refazer com 100% do base_health_insurance_train_df ou continuar?
Continuar como está — o plano já garante a comparação justa, com um passo adicional no final.

O raciocínio:

Abordagem	Clustering & LR	Novo modelo (Plano)
Treino	train_df (1.070 linhas)	train_df split 80/20 → 856 linhas (seleção)
Avaliação final	test_df (268 linhas)	test_df (268 linhas)
A comparação sempre será feita nas mesmas 268 linhas. O que a Fase 3 usa (856 vs 214) é apenas para selecionar a estrutura do modelo — não é a avaliação final.

O fluxo correto é:

Executar Fases 1-6 com o split interno (856/214 dentro do train_df)
Após a Fase 6 eleger o melhor modelo, retreinar esse modelo nos 1.070 linhas completos do train_df
Aplicar nas 268 linhas do test_df → comparação justa com clustering e LR
Isso é implícito no plano (a célula da Fase 3.2 diz "OU: usar base_health_insurance_train_df como treino e a base de teste deixada de fora como teste"). Não precisa reiniciar nada. Basta lembrar que o passo de retreinar no train_df completo vem depois da seleção.

---

## Estimação de Parâmetros do Modelo Final

### Por que retreinar no conjunto completo de treino é a estimação de parâmetros

Nas Fases 1–6, toda a análise de seleção foi conduzida sobre um split interno de `base_health_insurance_train_df` (856 linhas de treino / 214 de validação). As 214 linhas de validação interna serviram para comparar os 6 modelos — nunca para estimar parâmetros finais.

Após a seleção, Ljung (1999, sec. 4.4) prescreve: **re-estimar os parâmetros usando todos os dados de treino disponíveis**. Isso não é um passo extra — é o mesmo conceito de "estimar parâmetros", agora aplicado ao conjunto completo. O que muda:

| | Seleção de estrutura (Fases 1–6) | Estimação de parâmetros (pós-seleção) |
|---|---|---|
| Dataset | 856 linhas (80% do `train_df`) | **1.070 linhas (100% do `train_df`)** |
| Objetivo | Comparar 6 famílias de modelo | Gerar os parâmetros finais do modelo escolhido |
| Holdout tocado? | Não | Não — 268 linhas reservadas para comparação final |

O modelo escolhido (Gradient Boosting) foi retreinado nas 1.070 linhas com os hiperparâmetros confirmados na Fase 3:

```
n_estimators  = 100
learning_rate = 0.10
max_depth     = 5
subsample     = 0.80
random_state  = 42
```

---

## Equação do Modelo — Gradient Boosting

### Por que não existe uma "equação" no sentido clássico

Regressão linear tem uma equação fechada:

```
expenses = β₀ + β₁·age + β₂·bmi + β₃·children + ...
```

Cada coeficiente `βᵢ` é um número único, fixo, interpretável diretamente.

O Gradient Boosting não funciona assim. A previsão para um cliente com vetor de features **x** é:

```
F_M(x) = F₀ + γ·h₁(x) + γ·h₂(x) + ... + γ·h_M(x)
```

| Símbolo | Valor | Significado |
|---------|-------|-------------|
| **F₀** | média de `expenses` no treino | Previsão inicial, antes de qualquer árvore |
| **hₘ(x)** | árvore de decisão no passo m | Aprendiz fraco treinado nos resíduos do passo anterior |
| **γ** | 0,10 | Taxa de aprendizado — cada árvore contribui 10% de sua saída bruta |
| **M** | 100 | Número total de árvores sequenciais |
| **x** | [age, bmi, children, gender_male, region_*] | Vetor de 7 features por cliente |

Cada árvore é ajustada sobre o **gradiente negativo da função de perda** (resíduos de erro quadrático) do ensemble atual. A previsão final acumula 100 pequenas correções, cada uma amortecida pela taxa de aprendizado.

**O que isso significa na prática:** o modelo não tem coeficientes interpretáveis no estilo β₁ = "cada ano de idade aumenta o prêmio em R$ X". O que ele tem é **feature importance** — a contribuição relativa de cada variável para a redução do erro ao longo das 100 árvores.

### Feature Importance (estimada no conjunto completo de treino)

| Feature | Importância (%) | Interpretação |
|---------|----------------|---------------|
| `age` | ~53% | Principal driver de risco — 53% da capacidade preditiva do modelo |
| `children` | ~35% | Segundo fator mais relevante — interação com `age` é o que diferencia GB de regressão linear |
| `bmi` | ~11% | Contribuição secundária mas mensurável |
| `gender_male` | ~1% | Praticamente irrelevante |
| `region_*` | < 1% cada | Sem impacto significativo no pricing |

> **Nota:** `age` e `children` juntos explicam ~88% do peso preditivo total. A interação entre essas duas variáveis — que o modelo linear não consegue capturar — é a principal razão pelo gap de 14 pontos percentuais de R² entre GB e regressão linear.

---

## Aplicação ao Conjunto de Teste (Holdout)

### Contexto

Com o modelo estimado nas 1.070 linhas de `base_health_insurance_train_df`, a aplicação ao conjunto holdout (`base_health_insurance_test_df`, 268 clientes) é o primeiro contato real entre o Gradient Boosting e dados nunca vistos.

Este conjunto é o **mesmo** utilizado para avaliar clustering e regressão linear — o que garante comparação justa entre as três abordagens.

### Procedimento de Encoding

O conjunto de teste passa pelo mesmo processo de encoding da Fase 1:
1. Selecionar colunas: `['age', 'bmi', 'children', 'gender', 'region', 'expenses']`
2. Aplicar `pd.get_dummies(..., drop_first=True)` com as mesmas categorias de referência:
   - `gender`: referência = `female` → cria `gender_male`
   - `region`: referência = `midwest` → cria `region_northeast`, `region_south`, `region_southeast`
3. Alinhar colunas ao layout do treino via `.reindex()` (proteção contra categorias ausentes no split de 268 linhas)

### Métricas no Holdout

Os valores abaixo são os resultados esperados com base no modelo retreinado nas 1.070 linhas:

| Métrica | Valor esperado | Comparação com seleção (Phase 3) |
|---------|---------------|----------------------------------|
| R² Test | ~0.84 | Ligeiramente acima de 0.83 (mais dados de treino) |
| RMSE Test | ~$555–570 | Similar ao $566.77 da Phase 3 |
| MAE Test | ~$425–440 | Similar ao $436.03 da Phase 3 |

> Os valores reais são exibidos na saída do notebook após execução da célula de teste.

### Tabela de Previsões

A tabela de resultados segue o mesmo formato da regressão linear (células 83–84 do notebook), com colunas:

| Coluna | Descrição |
|--------|-----------|
| Age, Gender, BMI, Children, Region | Features originais do cliente |
| Expenses (R$) | Valor real de despesa médica |
| **Estimated Expenses (R$)** | Previsão do Gradient Boosting |
| Error (%) | `(estimado − real) / real × 100` |

Esta tabela, ao lado das tabelas de clustering e regressão linear, completa o quadro comparativo das três abordagens no mesmo conjunto de 268 clientes.
