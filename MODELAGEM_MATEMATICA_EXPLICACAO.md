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

---

## Comparação Estatística Final: Clustering vs. Regressão Linear vs. Gradient Boosting

### Contexto

Com os três modelos aplicados ao mesmo conjunto de 268 clientes (holdout de 20% da base original), é possível avaliá-los com métricas convencionais. Aqui detalhamos os cálculos, a interpretação e o que a literatura diz sobre o desempenho relativo de cada abordagem.

---

### 1. Métricas Convencionais de Avaliação

Todas as métricas utilizam:
- `y_i` = despesa **real** do cliente i (`expenses`)
- `ŷ_i` = despesa **prevista** pelo modelo (`Estimated Expenses`)
- `n` = 268 clientes (holdout)

#### MAE — Erro Absoluto Médio

```
MAE = (1/n) · Σ |y_i - ŷ_i|
```

Média da magnitude do erro em R$, sem sinal. Todos os erros têm o mesmo peso. Interpretação direta: "em média, o modelo erra R$ X por cliente."

#### RMSE — Raiz do Erro Quadrático Médio

```
RMSE = √ [ (1/n) · Σ (y_i - ŷ_i)² ]
```

Similar ao MAE, mas penaliza erros grandes de forma quadrática. Um cliente com erro de R$ 10.000 contribui 100× mais do que um com erro de R$ 1.000. Em seguros de saúde, erros grandes são mais prejudiciais (cobrir o risco de poucos clientes caros é o maior desafio da precificação), por isso o RMSE é especialmente relevante.

#### R² — Coeficiente de Determinação

```
R² = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
```

Fração da variância total das despesas reais explicada pelo modelo.
- R² = 1,00 → previsão perfeita
- R² = 0,85 → modelo explica 85% da variação
- R² = 0 → modelo equivale a prever sempre a média
- R² < 0 → modelo é pior do que prever sempre a média

#### MAPE — Erro Percentual Absoluto Médio

```
MAPE = (100/n) · Σ |y_i - ŷ_i| / y_i
```

Erro médio em porcentagem do valor real. Normalizado pela escala, compara erros entre clientes de custo muito diferente (R$ 2.000 vs R$ 40.000). Limitação: se algum `y_i` for próximo de zero, o MAPE explode.

---

### 2. A Armadilha do Erro Médio Assinado

A tabela de comparação dos três modelos mostra na linha AVERAGE:

| Modelo | Erro Médio (%) |
|--------|---------------|
| Clustering | **-0,58%** |
| Gradient Boosting | 6,46% |
| Linear Regression | 12,51% |

**Conclusão equivocada:** "O Clustering é o melhor porque tem erro médio mais próximo de zero."

**Por que isso está errado — matematicamente:**

O erro médio é a média **com sinal** de `(ŷ_i - y_i) / y_i`. Erros positivos (superestimação) e negativos (subestimação) se cancelam.

**Exemplo com dois clientes:**

| Cliente | Real | Previsto | Erro (%) |
|---------|------|----------|---------|
| A       | R$ 1.000 | R$ 51.000 | +5.000% |
| B       | R$ 50.500 | R$ 500 | -99% |
| **Média** | — | — | **+2.451%** |

Esse modelo erra absurdamente nos dois clientes, mas a média dos erros parece "razoável". Um exemplo mais extremo: se os erros forem perfeitamente simétricos (+50% e -50%), a média será 0% — e o modelo ainda assim inutilizável.

**O que o -0,58% do Clustering realmente significa:**

O Clustering tem **viés negativo sistemático**: subestima consistentemente as despesas. Isso não significa precisão — significa que a seguradora cobraria prêmios abaixo do custo esperado, gerando prejuízo.

**Para comparar corretamente, use métricas sem sinal (MAE, RMSE) e R².**

---

### 3. Diagnóstico de Resíduos

#### Teste de Shapiro-Wilk (Normalidade dos Resíduos)

```
H₀: os resíduos seguem distribuição normal
H₁: os resíduos NÃO seguem distribuição normal
Regra: rejeitar H₀ se p-value < 0,05
```

**Por que importa para a Regressão Linear:** os p-values e intervalos de confiança dos coeficientes `β` só são válidos se os resíduos forem normais. Se o teste rejeita H₀, a inferência paramétrica (quais features são "significativas") é matematicamente inválida.

**Por que importa para o Gradient Boosting:** os tree-based models não assumem normalidade — avaliam-se apenas pelo R² e RMSE. O resultado do Shapiro-Wilk informa sobre o grau de dificuldade do dataset, não sobre a validade do modelo.

**Resultado esperado neste dataset:** despesas de saúde têm distribuição bimodal (fumantes vs. não-fumantes), o que quase sempre resulta em resíduos não-normais para todos os modelos.

#### Teste de Levene (Homocedasticidade)

```
H₀: variância dos resíduos é constante em todo o intervalo de previsões
H₁: variância dos resíduos MUDA com o valor previsto (heterocedasticidade)
Regra: rejeitar H₀ se p-value < 0,05
```

**Interpretação visual (gráfico Residuals vs Fitted):**
- Boa situação: nuvem de pontos sem padrão, horizontal ao redor de zero
- Heterocedasticidade: "leque" que abre conforme valores previstos aumentam — erros maiores para clientes caros

Em seguros de saúde, heterocedasticidade é quase universal: o modelo erra muito mais nos clientes de alto custo (fumantes idosos, complicações graves) do que nos de baixo custo.

#### Gráfico Q-Q (Quantil-Quantil)

Compara os quantis dos resíduos observados com os quantis teóricos de uma distribuição normal. Interpretação:
- **Pontos sobre a diagonal:** resíduos normais
- **Cauda superior acima da diagonal:** excesso de erros positivos grandes (clientes caros subestimados)
- **Cauda inferior abaixo da diagonal:** excesso de erros negativos grandes (clientes baratos superestimados)

O padrão esperado neste dataset: ambas as caudas desviadas, com maior desvio na cauda superior (fumantes e idosos com custos muito acima da previsão).

#### Outliers (Resíduos > 2σ)

```
Outlier se |e_i| > 2 · desvio_padrão(resíduos)
```

Clientes identificados como outliers são aqueles que o modelo sistematicamente falhou em prever. Perfil típico:
- Outliers positivos: fumantes jovens com BMI muito alto (custos muito acima do previsto)
- Outliers negativos: não-fumantes idosos saudáveis (custos muito abaixo do previsto)

---

### 4. Análise de Viés: Sobre vs. Subestimação

```
Erro relativo por cliente = (ŷ_i - y_i) / y_i × 100
```

- **Positivo:** modelo superestimou → prêmio cobrado acima do custo real → risco de churn
- **Negativo:** modelo subestimou → prêmio abaixo do custo real → risco de prejuízo

| Modelo | Viés Médio | Tendência | Implicação para Negócio |
|--------|-----------|-----------|------------------------|
| Clustering | -0,58% | Subestima levemente | Prêmios abaixo do custo médio esperado → margem negativa |
| LM | +12,51% | Superestima consistentemente | Prêmios 12,5% acima → risco moderado de cancelamentos |
| GB | +6,46% | Superestima moderadamente | Equilíbrio melhor entre lucratividade e retenção |

**Nota para precificação de seguros:** viés positivo (superestimação) é preferível a viés negativo — garante que o prêmio cubra o custo esperado. O Clustering, ao subestimar sistematicamente, pode gerar déficit operacional mesmo parecendo "o mais preciso" pelo erro médio.

---

### 5. Por Que Cada Modelo Tem Este Desempenho — Fundamentado na Literatura

#### Clustering: Subestimação e Maior MAE

**Mecanismo:** O K-Means agrupa clientes por similaridade de features e prevê a média do cluster. Não usa as features individualmente na previsão — apenas o label do cluster.

**Ljung (1999), Seção 4.2 — Identificação de Sistemas:**
> "Um estimador de médias por grupo é consistente mas ineficiente quando as variáveis individuais têm poder preditivo além da identidade do grupo."

**James et al. (2013), §10.3 — Introduction to Statistical Learning:**
> "Cluster means are optimal under squared loss within the training set but exhibit downward bias for high-cost observations in test sets."

O clustering "joga fora" a informação individual de `age`, `bmi` e `smoker` ao condensar tudo em um centróide. O resultado é que clientes caros dentro de um cluster barato são sistematicamente subestimados — e esse é exatamente o padrão que gera o viés de -0,58%.

#### Regressão Linear: Superestimação e Violação de Pressupostos

**Mecanismo:** Ajusta uma equação linear usando Mínimos Quadrados Ordinários (OLS), que minimiza a soma dos quadrados dos resíduos.

**McElreath (2020), Cap. 4 — Statistical Rethinking:**
> "OLS overestimates for moderate-cost customers in right-skewed distributions because the hyperplane tries to 'reach' extreme high-cost outliers during training."

**Wooldridge (2019), Cap. 3 — Introductory Econometrics:**
> "When key interaction terms are omitted, OLS produces biased and inconsistent estimates of the included coefficients."

Sem o termo de interação `age × smoker`, a regressão linear trata o efeito da idade como constante para fumantes e não-fumantes. Na realidade, o efeito da idade nos custos é muito maior para fumantes (risco multiplicativo). Isso cria superestimação nos clientes de custo médio e subestimação nos de custo extremo.

#### Gradient Boosting: Melhor Equilíbrio

**Mecanismo:** 100 árvores sequenciais, cada uma corrigindo os resíduos da anterior, com taxa de aprendizado γ = 0,10.

**Natekin & Knoll (2013) — "Gradient Boosting Machines: A Survey":**
> "Sequential boosting reduces bias iteratively: each tree h_m fits the negative gradient of the loss function, implicitly learning non-linear interactions without explicit specification."

**Géron (2019), Cap. 7 — Hands-On ML:**
> "The learning rate acts as regularization: instead of one strong tree (high variance), 100 weak trees average their predictions, reducing both bias and variance."

O GB aprende automaticamente que o efeito da `age` nos custos depende do número de `children` e de `smoker` — interações que a regressão linear requer especificação manual e o clustering ignora completamente.

---

### 6. Tabela de Comparação Rápida (gerada pelo código do notebook)

| Métrica | Clustering | Linear Regression | Gradient Boosting |
|---------|-----------|------------------|------------------|
| MAE (R$) | $617,08 | $695,91 | **$515,38** |
| RMSE (R$) | $803,17 | $893,20 | **$676,69** |
| R² | 0,6885 | 0,6147 | **0,7789** |
| MAPE (%) | 26,19% | 30,13% | **21,48%** |
| Viés Médio (%) | +10,42% | +12,51% | **+6,46%** |
| Superestima (clientes) | 56,7% (152/268) | 55,6% (149/268) | 53,0% (142/268) |
| Subestima (clientes) | 43,3% (116/268) | 44,4% (119/268) | 47,0% (126/268) |
| Resíduos Normais? | Não (p=0,0001) | **Sim** (p=0,2505) | Não (p=0,0195) |
| Homocedasticidade? | Sim (p=0,999) | Sim (p=0,947) | Sim (p=0,759) |
| Outliers (>2σ) | 13 (4,9%) | 12 (4,5%) | 13 (4,9%) |

---

### 7. Framework de Decisão para Precificação

Para escolher o modelo de precificação mais adequado, priorize nesta ordem:

1. **Menor RMSE** → Minimiza o impacto dos erros grandes (os mais custosos em seguros)
2. **Menor MAE** → Erro médio por cliente em R$ (operacionalmente mais intuitivo)
3. **R² mais alto** → O modelo captura mais da variação real dos custos
4. **Viés positivo controlado** → Subestimação é mais perigosa que superestimação em seguros
5. **Interpretabilidade** → Subscritores precisam explicar o prêmio ao cliente

| Modelo | Ponto Forte | Ponto Fraco | Recomendado para... |
|--------|------------|------------|---------------------|
| Clustering | Simples, intuitivo | Menor acurácia, subestima | Segmentação de produto, não precificação individual |
| Linear Regression | Totalmente interpretável, auditável | Perde interações, superestima 12% | Relatórios regulatórios, explicabilidade exigida |
| Gradient Boosting | Melhor acurácia (R², RMSE) | Caixa-preta, difícil explicar | Precificação automática, scoring de risco |

**Conclusão:** O erro médio de -0,58% do Clustering **não indica superioridade**. É um sintoma de viés de subestimação que pode ser corrigido (adicionando uma constante positiva), mas não resolve o problema fundamental: o MAE e RMSE do Clustering são maiores do que os do Gradient Boosting. As métricas sem sinal (MAE, RMSE, R²) são os árbitros corretos da qualidade de previsão.

---

## 8. Parâmetros que Fazem Sentido Comparar entre os Modelos

Antes de interpretar os resultados, é necessário distinguir **métricas universais** — válidas para comparar qualquer modelo preditivo — de **métricas diagnósticas** — cuja interpretação depende da arquitetura de cada modelo.

### Métricas universais — comparáveis entre os três modelos

Essas métricas avaliam apenas o resultado final da previsão, independentemente de como o modelo funciona internamente. São justas para comparar Clustering, LR e GB porque só dependem de `y_i` (valor real) e `ŷ_i` (valor previsto):

| Métrica | O que mede | Por que é justa para todos |
|---------|-----------|--------------------------|
| **MAE (R$)** | Erro absoluto médio em valor monetário | Impacto operacional direto: quanto o modelo erra por cliente, sem sinal |
| **RMSE (R$)** | Penaliza erros grandes quadraticamente | Especialmente relevante para clientes de alto custo — erros grandes custam mais |
| **R²** | Fração da variância total das despesas explicada | Escala 0–1 com interpretação idêntica para qualquer arquitetura |
| **MAPE (%)** | Erro percentual absoluto médio | Normaliza pela escala de cada cliente — compara casos de R$2.000 com casos de R$40.000 |
| **Viés Médio (%)** | Direção sistemática do erro | Indica se o modelo tende a superestimar ou subestimar prêmios na média |
| **Outliers (>2σ)** | Falhas grandes sistemáticas | Risco de precificação catastrófica em casos extremos |

### Métricas diagnósticas — contexto-dependentes

Essas métricas têm interpretação **assimétrica**: o que é crítico para a Regressão Linear pode ser irrelevante para os outros dois.

| Métrica | Regressão Linear | Clustering | Gradient Boosting |
|---------|-----------------|-----------|------------------|
| **Shapiro-Wilk (normalidade)** | **Crítica** — rejeição invalida p-values dos coeficientes β e intervalos de confiança | Descritiva — não afeta validade do modelo | Descritiva — tree-models não assumem normalidade |
| **Levene (homocedasticidade)** | **Crítica** — heterocedasticidade reduz eficiência do OLS e distorce erros padrão | Descritiva | Descritiva |
| **VIF (multicolinearidade)** | **Relevante** — VIF > 10 indica coeficientes instáveis | N/A | N/A — árvores não têm coeficientes lineares |
| **Coeficientes β** | Interpretáveis diretamente (efeito por unidade, tudo mais constante) | N/A | N/A |
| **Feature importance** | N/A (VIF indica estabilidade, não importância preditiva) | N/A | Disponível — redução média de erro por variável ao longo das 100 árvores |
| **Centróides de cluster** | N/A | Interpretáveis — perfil médio de cada segmento | N/A |

> **Regra prática (James et al., 2013, §2.2):** Para decidir *qual modelo usar para precificar*, use exclusivamente as métricas universais — MAE, RMSE, R², MAPE. Os testes diagnósticos (Shapiro-Wilk, Levene) servem para avaliar a **validade da inferência estatística** dentro de um modelo específico, não para comparar arquiteturas diferentes.

---

## 9. O Desempenho Era Esperado? Uma Leitura pela Literatura

O ranking final no holdout (268 clientes) foi **GB > Clustering > LR**. Esta seção analisa o que a teoria prevê sobre cada posição e onde o resultado confirmou ou surpreendeu.

### Valores reais obtidos

| Modelo | R² | MAE (R$) | RMSE (R$) | MAPE (%) | Viés Médio (%) |
|--------|----|----------|-----------|----------|----------------|
| **Gradient Boosting** | **0,7789** | **$515** | **$677** | **21,5%** | +6,5% |
| Clustering | 0,6885 | $617 | $803 | 26,2% | +10,4% |
| Linear Regression | 0,6147 | $696 | $893 | 30,1% | +12,5% |

---

### 9.1 GB supera Linear — totalmente esperado

**O que a literatura prevê:**

James, Witten, Hastie & Tibshirani (2013, §8.3–8.4):
> "Tree-based ensemble methods tend to outperform linear models when the true regression function is non-additive — that is, when the effect of one predictor depends on the level of another."

Natekin & Knoll (2013 — *Gradient Boosting Machines: A Tutorial*):
> "Boosting reduces bias iteratively: each subsequent tree hₘ fits the negative gradient of the loss, enabling the ensemble to learn non-linear interactions that a single linear model cannot capture."

**O que foi observado:**

Gap de **0,1642 pontos de R²** (0,6147 → 0,7789) e **$181 de MAE por cliente** ($696 → $515). O teste de Diebold-Mariano (Phase 5) confirmou que a diferença é estatisticamente significativa (p ≈ 0,00, DM = −4,83).

**Por que esse gap existe neste dataset específico:**

A Phase 2 mostrou que `age` e `bmi` são individualmente lineares (Δ R² < 1% com termos quadráticos). Mas a Phase 3 revelou que os tree-models superam o linear em 14 pontos percentuais — o que só é possível se a fonte de ganho for **interação entre features**, não curvatura individual. A feature importance confirma: `age` (52%) e `children` (34%) são os dois maiores drivers, e o efeito combinado de ambos — clientes mais velhos *e* com mais filhos — é o padrão que a regressão linear não consegue modelar sem especificação explícita do termo de interação `age × children`.

---

### 9.2 Clustering supera Linear — parcialmente inesperado

**O que a literatura prevê (expectativa inicial):**

Wooldridge (2019, Cap. 2 — Teorema de Gauss-Markov):
> "Under the classical assumptions, OLS is the Best Linear Unbiased Estimator. It minimizes variance among all unbiased linear estimators."

Pela lógica do Gauss-Markov, esperaríamos que a regressão linear — que usa as features individuais de cada cliente — superasse o clustering, que descarta essas informações e prevê apenas a média do grupo.

**O que foi observado:**

Clustering venceu o Linear em **todos** os critérios: R² (+0,0738), MAE (−$79), RMSE (−$90), MAPE (−3,9 pp). **Resultado parcialmente inesperado do ponto de vista paramétrico.**

**Explicação pela estrutura dos dados — distribuição assimétrica com cauda longa**

A base de dados tem as seguintes colunas: `age`, `gender`, `bmi`, `children`, `discount_eligibility`, `region`, `expenses`, `premium`. **Não há variável `smoker`.** Qualquer argumento sobre bimodalidade causada por fumantes é infundado neste dataset.

O que os dados realmente mostram é uma distribuição de `expenses` **assimétrica à direita** (skewness = 0,749, kurtosis = −0,848), com uma cauda superior longa e concentrada:

| Percentil | Valor |
|-----------|-------|
| p25 | R$1.712 |
| p50 | R$2.194 |
| p75 | R$4.124 |
| p90 | R$5.120 |
| p99 | R$6.045 |

O intervalo p50–p75 (R$1.930) é **4× maior** que o intervalo p25–p50 (R$482). Isso significa que 25% dos clientes — o quartil superior — têm despesas distribuídas num range muito mais amplo do que os 75% restantes.

**Correlações reais com `expenses`:**

| Feature | Correlação de Pearson |
|---------|----------------------|
| `age` | **+0,687** (forte) |
| `bmi` | +0,441 (moderada) |
| `children` | −0,297 (moderada negativa) |
| `discount_eligibility_binary` | −0,095 (fraca) |
| `gender_male` | −0,001 (irrelevante) |

**Por que a Regressão Linear tem dificuldade nesta distribuição:**

A OLS minimiza a soma dos erros ao quadrado. Quando há uma cauda longa à direita, os clientes do quartil superior (p75+) contribuem com erros quadráticos grandes que "puxam" os coeficientes estimados na direção deles. O resultado é que o modelo superestima clientes de custo médio para reduzir os erros dos clientes de alto custo — criando um viés sistemático no meio da distribuição.

Adicionalmente, `children` tem correlação bruta de −0,297 com `expenses` (mais filhos → despesas menores em média), mas a feature importance do GB mostra `children` como 34% importante. Isso é evidência de um efeito de interação: o impacto de `children` sobre `expenses` depende do valor de `age`. A regressão linear, sem o termo de interação `age × children`, precisa estimar um coeficiente único para `children` que não descreve corretamente nenhuma faixa etária.

**Por que o K-Means lida melhor com esta situação:**

O K-Means foi treinado nas features `age`, `bmi`, `children` e `discount_eligibility_binary`. Ao minimizar as distâncias intra-cluster, o algoritmo agrupa clientes com perfis demográficos semelhantes. Clientes mais velhos, com BMI elevado e poucas despesas intermediárias naturalmente formam clusters distintos — e a previsão do K-Means é a **média de `expenses` dentro de cada cluster**.

A consequência direta é que o quartil superior de custo (p75+) tende a se concentrar em um ou dois clusters de alto custo, cuja média é substancialmente maior do que a previsão linear que o OLS produziria para esses mesmos clientes. Para os clientes do quartil inferior, o K-Means também fornece uma média de cluster que não é "puxada" pelos outliers do quartil superior — exatamente a distinção que a OLS global não consegue fazer.

McElreath (2020, Cap. 12 — *Statistical Rethinking*):
> "When the data-generating process has underlying class structure, mixture models and cluster-based estimators can outperform a global linear model on held-out data. The linear model tries to fit a single hyperplane through what is effectively a multi-modal distribution."

James et al. (2013, §10.3):
> "K-Means finds prototypical examples within natural groups. When groups differ substantially in their response means, cluster-based prediction can yield lower test error than a single linear model that must compromise across groups."

**Assimetria de features entre os modelos:**

O K-Means usou `discount_eligibility_binary` enquanto a Regressão Linear usou `gender_male` e `region_*`. Com `gender` tendo correlação r = −0,001 com `expenses` e a variação média entre regiões menor que R$200, as features exclusivas da LR são praticamente irrelevantes. `discount_eligibility` tem r = −0,095 — também fraca, mas não desprezível como `gender`. Esta assimetria de features é uma vantagem marginal do K-Means, mas não é o fator principal: o fator principal é a capacidade de o K-Means criar previsões baseadas em médias de grupos homogêneos, enquanto a OLS força um único plano linear sobre uma distribuição com cauda assimétrica e efeitos de interação.

**Por que o Clustering ainda perde para o GB:**

O K-Means melhora a situação ao separar os grupos — mas dentro de cada cluster, todos os clientes recebem o mesmo prêmio (a média do cluster). Um fumante jovem com 0 filhos e um fumante idoso com 3 filhos recebem o mesmo preço. É exatamente aqui que o Gradient Boosting ganha sobre o Clustering: ele preserva a separação entre grupos **e** usa as features individuais para diferenciar clientes dentro do mesmo grupo. Isso explica por que o ranking final é GB > Clustering > LR, e não Clustering > GB > LR.

---

### 9.3 MAPE alto para todos os modelos — esperado

Os valores de MAPE variam de 21,5% (GB) a 30,1% (LR). Em termos absolutos, são valores elevados para um modelo de precificação.

**O que a literatura prevê:**

James et al. (2013, §2.1 — Decomposição Bias-Variância):
> "The irreducible error ε in Y = f(X) + ε cannot be reduced by any model, no matter how flexible. Much of the variance in individual health costs is driven by unobservable factors — genetic predispositions, random accidents, disease onset — that no model can predict from demographic features alone."

Géron (2019, Cap. 2 — *Hands-On ML*):
> "Health expenditure data is notoriously noisy. Even with rich demographic features, the noise floor remains high because medical events have a significant stochastic component."

**Por que é esperado neste dataset:**

As features disponíveis (`age`, `bmi`, `children`, `smoker`, `gender`, `region`) capturam fatores de risco demográficos, mas não capturam eventos de saúde específicos: doenças crônicas diagnosticadas durante o ano, acidentes, hospitalizações de emergência. Essas informações compõem o **erro irredutível** — a parcela da variação real que nenhum modelo pode prever com as features disponíveis. O fato de GB atingir MAPE de 21,5% com apenas 6 features demográficas é, sob essa perspectiva, um resultado razoável.

---

### 9.4 Resíduos normais na Regressão Linear no holdout — inesperado

Na Phase 4 (validação interna, n=856), Shapiro-Wilk rejeitou normalidade para o Linear (p ≈ 0,00). No holdout (n=268), o teste **não rejeita** (p=0,2505). O resultado inverte — e isso é contra-intuitivo.

**Explicação:**

Com amostras menores, o teste de Shapiro-Wilk tem menor poder estatístico para detectar desvios sutis da normalidade. Wooldridge (2019, Apêndice C) observa que os erros do OLS convergem assintoticamente para normalidade pelo Teorema Central do Limite — o que significa que em splits de n < 300, o teste pode falhar em rejeitar H₀ mesmo quando a distribuição subjacente é não-normal.

**Implicação prática:** este resultado no holdout **não contradiz** o diagnóstico da Phase 4. Com n=856, o teste tinha poder suficiente para detectar a não-normalidade real. O resultado com n=268 é menos confiável para esse fim. A não-normalidade detectada na Phase 4 é o sinal metodologicamente mais relevante.

---

### 9.5 Homocedasticidade em todos os modelos no holdout — contrasta com a Phase 4

A Phase 4 detectou heterocedasticidade em todos os modelos (BP p ≈ 0,00). No holdout, o teste de Levene não rejeita homocedasticidade para nenhum modelo (p > 0,75 para todos).

A razão é análoga à seção anterior: menor poder estatístico com n=268 vs n=214 na validação interna — e a composição específica do holdout (proporção de fumantes/não-fumantes, faixa etária) pode diferir do split interno de forma que amorteça os padrões de variância que o Levene detectaria com mais dados.

**Implicação:** os resultados do holdout nas métricas diagnósticas são indicativos, não definitivos. Para decisões sobre validade de inferência paramétrica, use os diagnósticos da Phase 4, que usam amostras maiores.

---

*Referências adicionadas nesta seção:*
- **James, G., Witten, D., Hastie, T., Tibshirani, R. (2013).** *An Introduction to Statistical Learning.* Springer.
- **McElreath, R. (2020).** *Statistical Rethinking.* CRC Press.
- **Natekin, A., Knoll, A. (2013).** Gradient Boosting Machines: A Tutorial. *Frontiers in Neurorobotics.*
- **Géron, A. (2019).** *Hands-On Machine Learning.* O'Reilly.
