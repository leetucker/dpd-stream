# An independent check of the bootstrap's recentered refits (DPDResult.refit_bootstrap)
# with momentfit (Chausse), a general GMM package: tests/test_momentfit.py compares
# dpd-stream with what this writes, tests/data/momentfit_recenter.json.
#
#   python benchmarks/external/momentfit_recenter_export.py
#   R_LIBS_USER=... Rscript benchmarks/external/momentfit_recenter_check.R
#
# For each case, from each unit's rows of y, X and Z (a unit is a panel, or a
# cluster):
#   1. the original sample's two-step (or iterated) GMM, with momentfit, its first
#      step weighted by (sum_i Z_i'HZ_i)^-1 and its second by the optimal weight
#      (the uncentered covariance of the moments, as xtabond2); the moment sum at
#      its estimate, g = sum_i Z_i'u_i, is the shift;
#   2. the draw's GMM with Hall and Horowitz's (1996) recentered moments,
#      g_i(theta) - g / N, in every step; its Hansen statistic from momentfit's
#      specTest, with the weight the estimate used (as xtabond2);
#   3. Windmeijer's (2005) corrected variance of the draw's two-step estimate,
#      V2 + D V1 D' + D V2 + V2 D', with D the derivative of the two-step estimate
#      with respect to the first-step estimate its weight matrix is computed at,
#      here by central finite differences rather than by formula.

suppressMessages({
  library(momentfit)
  library(arrow)
  library(jsonlite)
})

dir <- file.path("benchmarks", "external", ".momentfit")
cases <- fromJSON(file.path(dir, "cases.json"), simplifyVector = TRUE)

load_sample <- function(case, tag, k, j) {
  d <- as.data.frame(read_parquet(file.path(dir, paste0(case, "_", tag, ".parquet"))))
  S1 <- as.matrix(read.csv(file.path(dir, paste0(case, "_", tag, "_S1.csv")),
                           header = FALSE))
  list(unit = d$unit, y = d$y, X = as.matrix(d[paste0("x", 0:(k - 1))]),
       Z = as.matrix(d[paste0("z", 0:(j - 1))]), S1 = unname(S1))
}

# each unit's moment, sum over its rows of z_r (y_r - x_r'theta), less shift / n;
# a unit without rows still has its row (-shift / n)
unit_moments <- function(theta, x) {
  e <- as.vector(x$y - x$X %*% theta)
  agg <- rowsum(x$Z * e, x$unit)
  M <- matrix(0, x$n, ncol(x$Z))
  M[as.integer(rownames(agg)) + 1, ] <- agg
  sweep(M, 2, x$shift / x$n)
}
unit_grad <- function(theta, x) -crossprod(x$Z, x$X) / x$n

fit_case <- function(D, n, shift, steps, k) {
  x <- c(D, list(n = n, shift = shift))
  model <- momentModel(unit_moments, x = x,
                       theta0 = setNames(rep(0, k), paste0("b", seq_len(k))),
                       grad = unit_grad, vcov = "iid", centeredVcov = FALSE)
  control <- list(reltol = 1e-15, maxit = 10000)
  fit1 <- gmmFit(model, weights = solve(D$S1), control = control)
  b1 <- coef(fit1)
  w2 <- evalWeights(model, theta = b1, w = "optimal")
  fit2 <- gmmFit(model, weights = w2, efficientWeights = TRUE, control = control)
  b2 <- coef(fit2)
  iterations <- 1
  if (steps == "iterated") {
    for (it in 1:100) {
      w <- evalWeights(model, theta = b2, w = "optimal")
      fit2 <- gmmFit(model, weights = w, efficientWeights = TRUE, control = control)
      bn <- coef(fit2)
      done <- max(abs(bn - b2)) <= 1e-9 * (1 + max(abs(b2)))
      b2 <- bn
      iterations <- it + 1
      if (done) break
    }
  }
  # J with the weight the estimate used (as xtabond2: for two-step, the weight at
  # the one-step estimate); specTest's default re-evaluates the weight at the
  # estimate itself, another form of the test, kept for the record
  J <- specTest(fit2, wObj = fit2@wObj)@test[1, 1]
  J_reweighted <- specTest(fit2)@test[1, 1]

  # the same in closed form (sums, not means), for the finite differences
  G <- crossprod(D$Z, D$X)
  m0 <- as.vector(crossprod(D$Z, D$y)) - shift
  S_at <- function(b) crossprod(unit_moments(b, x))
  b_of <- function(W) as.vector(solve(t(G) %*% W %*% G, t(G) %*% W %*% m0))
  W1 <- solve(D$S1)
  b1c <- b_of(W1)
  W2 <- solve(S_at(b1c))
  b2c <- b_of(W2)
  V2 <- solve(t(G) %*% W2 %*% G)
  A1 <- solve(t(G) %*% W1 %*% G, t(G) %*% W1)
  V1r <- A1 %*% S_at(b1c) %*% t(A1)
  Dm <- matrix(0, k, k)
  for (kk in seq_len(k)) {
    h <- 1e-5 * max(1, abs(b1c[kk]))
    up <- dn <- b1c
    up[kk] <- up[kk] + h
    dn[kk] <- dn[kk] - h
    Dm[, kk] <- (b_of(solve(S_at(up))) - b_of(solve(S_at(dn)))) / (2 * h)
  }
  Vw <- V2 + Dm %*% V1r %*% t(Dm) + Dm %*% V2 + V2 %*% t(Dm)
  if (steps == "iterated") {
    # uncorrected, at convergence (the weight of the last step's estimate)
    Vw <- solve(t(G) %*% solve(S_at(b2)) %*% G)
  }
  list(b1 = unname(b1), b = unname(b2), b1_closed = b1c, b2_closed = b2c,
       hansen = J, hansen_reweighted = J_reweighted, V = unname(Vw),
       V2 = unname(V2), iterations = iterations,
       moment_sum = as.vector(colSums(unit_moments(b2, x))))
}

out <- list()
for (case in names(cases)) {
  cs <- cases[[case]]
  k <- cs$k
  j <- cs$j
  orig <- load_sample(case, "orig", k, j)
  n_orig <- max(orig$unit) + 1
  o <- fit_case(orig, n_orig, rep(0, j), cs$steps, k)
  draw <- load_sample(case, "draw", k, j)
  d <- fit_case(draw, cs$n_units, o$moment_sum, cs$steps, k)
  out[[case]] <- list(original_b = o$b, shift = o$moment_sum, b1 = d$b1, b = d$b,
                      b1_closed = d$b1_closed, b2_closed = d$b2_closed,
                      hansen = d$hansen, hansen_reweighted = d$hansen_reweighted,
                      V = d$V, iterations = d$iterations,
                      draw_moment_sum_at_b = d$moment_sum)
  cat(case, ": b =", round(d$b, 6), " J* =", round(d$hansen, 4), "\n")
}
out$momentfit_version <- as.character(packageVersion("momentfit"))
write_json(out, file.path("tests", "data", "momentfit_recenter.json"), digits = NA,
           auto_unbox = TRUE, pretty = TRUE)
