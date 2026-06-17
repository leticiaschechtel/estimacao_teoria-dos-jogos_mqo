# ============================================================
# Aula aplicada de Econometria: MQO matricial e Copa do Mundo
# Modelo didático inspirado em Joachim Klement
# Dados simulados: NÃO são dados oficiais.
# ============================================================

# Objetivo da aula:
# 1) Estimar uma regressão múltipla por álgebra matricial.
# 2) Conferir o resultado com lm().
# 3) Fazer teste de contribuição incremental de uma variável.
# 4) Gerar um ranking previsto para as seleções de 2026.

# ------------------------------------------------------------
# 0. Leitura dos dados
# ------------------------------------------------------------

treino <- read.csv("copa_treino_simulada.csv", stringsAsFactors = FALSE)
copa2026 <- read.csv("copa_2026_simulada.csv", stringsAsFactors = FALSE)

# Visualização inicial
head(treino)
head(copa2026)

# ------------------------------------------------------------
# 1. Preparação das variáveis
# ------------------------------------------------------------

treino$lpop <- log(treino$pop_milhoes)
treino$lgdppc <- log(treino$pibpc_usd)
treino$lgdppc2 <- treino$lgdppc^2
treino$temp14 <- (treino$temp_media_c - 14)^2

copa2026$lpop <- log(copa2026$pop_milhoes)
copa2026$lgdppc <- log(copa2026$pibpc_usd)
copa2026$lgdppc2 <- copa2026$lgdppc^2
copa2026$temp14 <- (copa2026$temp_media_c - 14)^2

# Variável dependente:
# score = 1 grupos; 2 oitavas; 3 quartas; 4 semifinal; 5 vice; 6 campeão.
table(treino$score)
table(treino$fase)

# ------------------------------------------------------------
# 2. Modelo irrestrito por álgebra matricial
# ------------------------------------------------------------

# Modelo:
# score_i = b0 + b1*fifa_points_i + b2*lpop_i + b3*lgdppc_i
#           + b4*lgdppc2_i + b5*temp14_i + b6*host_i + u_i

y <- as.matrix(treino$score)

X_ur <- model.matrix(
  ~ fifa_points + lpop + lgdppc + lgdppc2 + temp14 + host,
  data = treino
)

# Estimador MQO: beta_chapeu = (X'X)^(-1)X'y
# Solve realiza a operação de inversão de matrizes, mas é mais eficiente e numéricamente estável do que usar solve(X'X) diretamente.'
# %% é para entender a multiplicação de matrizes, não é o operador de módulo.'
# beta_ur é um vetor coluna de estimativas dos parametros'.

beta_ur <- solve(t(X_ur) %*% X_ur) %*% t(X_ur) %*% y
beta_ur


# Valores ajustados e resíduos
y_hat_ur <- X_ur %*% beta_ur
u_ur <- y - y_hat_ur

# Soma dos quadrados dos resíduos
SQR_ur <- sum(u_ur^2)

# n e k
n <- nrow(X_ur)
k_ur <- ncol(X_ur)

# Variância estimada do erro
sigma2_ur <- SQR_ur / (n - k_ur)

# Matriz de variância-covariância dos betas
var_beta_ur <- sigma2_ur * solve(t(X_ur) %*% X_ur)

# Erros-padrão e estatísticas t
se_beta_ur <- sqrt(diag(var_beta_ur))
t_ur <- beta_ur / se_beta_ur

tabela_manual <- data.frame(
  variavel = rownames(beta_ur),
  coeficiente = as.vector(beta_ur),
  erro_padrao = as.vector(se_beta_ur),
  t_calculado = as.vector(t_ur)
)

tabela_manual_exibir <- tabela_manual
tabela_manual_exibir[, c("coeficiente", "erro_padrao", "t_calculado")] <-
  round(tabela_manual_exibir[, c("coeficiente", "erro_padrao", "t_calculado")], 4)

tabela_manual_exibir

# ------------------------------------------------------------
# 3. Conferência com lm()
# ------------------------------------------------------------

modelo_ur <- lm(
  score ~ fifa_points + lpop + lgdppc + lgdppc2 + temp14 + host,
  data = treino
)

summary(modelo_ur)

# Repare que os coeficientes do lm() são os mesmos da conta matricial.
coef(modelo_ur)
as.vector(beta_ur)

# ------------------------------------------------------------
# 4. Teste de contribuição incremental de UMA variável
# ------------------------------------------------------------

# Pergunta:
# O ranking FIFA acrescenta explicação ao desempenho,
# depois de controlar por população, PIB per capita, temperatura e sede?

# H0: beta_fifa_points = 0
# H1: beta_fifa_points != 0

# Modelo restrito: retira fifa_points
X_r <- model.matrix(
  ~ lpop + lgdppc + lgdppc2 + temp14 + host,
  data = treino
)

beta_r <- solve(t(X_r) %*% X_r) %*% t(X_r) %*% y
u_r <- y - X_r %*% beta_r
SQR_r <- sum(u_r^2)

# Número de restrições
q <- ncol(X_ur) - ncol(X_r)

# Estatística F
F_calc <- ((SQR_r - SQR_ur) / q) / (SQR_ur / (n - k_ur))
F_calc

# p-valor do teste F
p_valor_F <- pf(F_calc, df1 = q, df2 = n - k_ur, lower.tail = FALSE)
p_valor_F

# Comparação com t^2, pois q = 1
t_fifa <- tabela_manual$t_calculado[tabela_manual$variavel == "fifa_points"]
t_fifa
t_fifa^2
F_calc

# Interpretação automática
if (p_valor_F < 0.05) {
  print("Rejeitamos H0: fifa_points contribui incrementalmente para explicar o desempenho.")
} else {
  print("Não rejeitamos H0: não há evidência de contribuição incremental de fifa_points.")
}


# ------------------------------------------------------------
# 5. Previsão para 2026
# ------------------------------------------------------------

X_2026 <- model.matrix(
  ~ fifa_points + lpop + lgdppc + lgdppc2 + temp14 + host,
  data = copa2026
)

copa2026$score_previsto <- as.vector(X_2026 %*% beta_ur)

ranking_2026 <- copa2026[order(-copa2026$score_previsto), 
                         c("grupo", "selecao", "codigo", "score_previsto",
                           "fifa_points", "pop_milhoes", "pibpc_usd", 
                           "temp_media_c", "host")]

ranking_2026$score_previsto <- round(ranking_2026$score_previsto, 3)

ranking_2026

# Top 10 favoritos do modelo didático
head(ranking_2026, 10)

# ------------------------------------------------------------
# 7. Gráfico simples do ranking previsto
# ------------------------------------------------------------

top10 <- head(ranking_2026, 10)

barplot(
  top10$score_previsto,
  names.arg = top10$codigo,
  las = 2,
  main = "Top 10 seleções segundo o modelo MQO simulado",
  ylab = "Score previsto"
)

# ------------------------------------------------------------
# 8. Observações para discussão com a turma
# ------------------------------------------------------------

# 1. Os dados são simulados, então não devem ser interpretados como previsão real.
# 2. A variável score é ordinal, mas aqui usamos MQO por finalidade didática.
# 3. O teste F incremental permite comparar modelo restrito e irrestrito.
# 4. Quando há apenas uma restrição, F = t^2.
# 5. O modelo serve para discutir os limites da previsão em eventos com muita aleatoriedade.



##############################################################################
#### Logit ordenado ##
##############################################################################

# Transformar score em fator ordenado
treino$score_ord <- ordered(
  treino$score,
  levels = c(1, 2, 3, 4, 5, 6)
)

# Logit ordenado
library(MASS)

modelo_logit_ord <- polr(
  score_ord ~ fifa_points + lpop + lgdppc + lgdppc2 + temp14 + host,
  data = treino,
  method = "logistic",
  Hess = TRUE
)

summary(modelo_logit_ord)

'O ponto mais importante é a interpretação: 
no modelo ordenado, o coeficiente positivo de fifa_points, 
por exemplo, não significa “aumenta o 
score em tantas unidades”, como no MQO. Significa que o 
aumento nos pontos FIFA desloca a seleção para uma maior 
probabilidade de estar nas categorias superiores 
de desempenho.'

# Probabilidades previstas para as seleções de 2026
prob_2026 <- predict(
  modelo_logit_ord,
  newdata = copa2026,
  type = "probs"
)

prob_2026_tabela <- data.frame(
  selecao = copa2026$selecao,
  prob_2026
)

head(prob_2026_tabela)

colnames(prob_2026_tabela) <- c(
  "selecao",
  "prob_grupos",
  "prob_oitavas",
  "prob_quartas",
  "prob_semifinal",
  "prob_vice",
  "prob_campeao"
)

head(prob_2026_tabela)

ranking_campeao <- prob_2026_tabela[
  order(-prob_2026_tabela$prob_campeao),
]

head(ranking_campeao, 10)

ranking_campeao_percentual <- ranking_campeao

ranking_campeao_percentual[, -1] <- round(
  100 * ranking_campeao_percentual[, -1],
  2
)

head(ranking_campeao_percentual, 10)
