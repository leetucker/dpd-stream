# One fit with an R package, for benchmarks/_harness.py: read the Parquet file
# into memory (timed as "read", as the Python libraries' reading is), fit, ask
# for the standard errors, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations. The
# coefficients of y's lag, x and w are named as dpd-stream names them.
#
#   Rscript gmm.R '<spec as JSON>'     (see the benchmarks' external_command)
#
# spec$fit is one of
#   pgmm     plm::pgmm, two-step, Windmeijer-corrected (vcovHC)
#   pdynmc   pdynmc::pdynmc, two-step, corrected standard errors
# spec$system: system GMM (else difference); spec$collapse; spec$path,
# spec$T (the periods, for the lag limits), spec$threads.
#
# The model is the benchmarks' (dpd-stream's formula in benchmarks/_models.py):
# y on its first lag, x (predetermined: lags 1 and deeper as GMM-style
# instruments), w (strictly exogenous: an IV-style instrument) and year
# effects; y's lags 2 and deeper as GMM-style instruments.

spec <- jsonlite::fromJSON(commandArgs(TRUE)[1], simplifyVector = TRUE)

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]

suppressPackageStartupMessages(library(arrow))
baseline <- hwm_mb()
t0 <- now()
df <- as.data.frame(read_parquet(spec$path, col_select = c("id", "year", "y", "x", "w")))
t_read <- now() - t0

maxlag <- spec$T
coefs <- list()
t0 <- now()
if (spec$fit == "pgmm") {
  suppressPackageStartupMessages(library(plm))
  pd <- pdata.frame(df, index = c("id", "year"))
  fml <- as.formula(sprintf(
    "y ~ lag(y, 1) + x + w | lag(y, 2:%d) + lag(x, 1:%d) | w", maxlag, maxlag))
  fit <- pgmm(fml, data = pd, effect = "twoways", model = "twosteps",
              transformation = if (isTRUE(spec$system)) "ld" else "d",
              collapse = isTRUE(spec$collapse))
  se <- sqrt(diag(vcovHC(fit)))
  b <- coef(fit)
  coefs <- list("l(y, 1)" = unname(b[1]), "x" = unname(b["x"]), "w" = unname(b["w"]))
  n_obs <- sum(sapply(fit$residuals, function(r) sum(!is.na(r))))
} else if (spec$fit == "pdynmc") {
  suppressPackageStartupMessages(library(pdynmc))
  fit <- pdynmc(dat = df, varname.i = "id", varname.t = "year",
                use.mc.diff = TRUE, use.mc.lev = isTRUE(spec$system), use.mc.nonlin = FALSE,
                inst.collapse = isTRUE(spec$collapse), inst.stata = FALSE,
                include.y = TRUE, varname.y = "y", lagTerms.y = 1, maxLags.y = maxlag,
                include.x = TRUE, varname.reg.pre = "x", lagTerms.reg.pre = 0,
                maxLags.reg.pre = maxlag,
                fur.con = TRUE, fur.con.diff = TRUE, fur.con.lev = isTRUE(spec$system),
                varname.reg.fur = "w", lagTerms.reg.fur = 0,
                include.dum = TRUE, dum.diff = TRUE, dum.lev = isTRUE(spec$system),
                varname.dum = "year",
                w.mat = "iid.err", std.err = "corrected", estimation = "twostep",
                opt.meth = "none")
  b <- fit$coefficients
  if (is.list(b)) b <- b[[length(b)]]
  se <- fit$stderr
  coefs <- list("l(y, 1)" = unname(b["L1.y"]), "x" = unname(b["L0.x"]),
                "w" = unname(b["L0.w"]))
  n_obs <- NA
} else {
  stop(paste("unknown fit", spec$fit))
}
t_fit <- now() - t0

cat("RESULT", jsonlite::toJSON(list(
  t_read = t_read, t_fit = t_fit, peak_memory_mb = hwm_mb(), baseline_memory_mb = baseline,
  peak_disk_mb = 0, n_obs = n_obs, iterations = 2, converged = TRUE, coefs = coefs),
  auto_unbox = TRUE, digits = NA, na = "null"), "\n")
