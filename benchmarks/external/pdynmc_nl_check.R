# pdynmc's Ahn-Schmidt fits on a balanced panel, to compare with dpd-stream's nl="noserial"
# (see DESIGN_NOTES.md, "pdynmc as a second reference for nl()").
#
#   Rscript pdynmc_nl_check.R data.csv      (columns id, year, y, x, w)
#
# The model is y ~ l(y, 1) + x + w, gmm(y, 2:7) + gmm(x, 1:3) + iv(w), no
# period effects; the data valid for the nonlinear moments (no omitted time
# effects). pdynmc caps the number of nonlinear moments by its lag limit for y
# and starts one period later than dpd-stream (it requires a third difference
# of y), so it has one moment fewer.
suppressPackageStartupMessages(library(pdynmc))
args <- commandArgs(TRUE)
df <- read.csv(args[1])
run <- function(est, opt = "BFGS", wmat = "iid.err", ...) {
  pdynmc(dat = df, varname.i = "id", varname.t = "year",
    use.mc.diff = TRUE, use.mc.lev = FALSE, use.mc.nonlin = TRUE,
    inst.collapse = FALSE,
    include.y = TRUE, varname.y = "y", lagTerms.y = 1, maxLags.y = 7,
    include.x = TRUE, varname.reg.pre = "x", lagTerms.reg.pre = 0, maxLags.reg.pre = 3,
    fur.con = TRUE, fur.con.diff = TRUE, fur.con.lev = FALSE,
    varname.reg.fur = "w", lagTerms.reg.fur = 0,
    include.dum = FALSE, w.mat = wmat, std.err = "corrected", estimation = est,
    opt.meth = opt, ...)
}
getb <- function(f) { b <- f$coefficients; if (is.list(b)) b <- b[[length(b)]]; unname(b) }
for (est in c("onestep", "twostep")) {
  f <- tryCatch(run(est, optCtrl = list(trace = FALSE, maxit = 500, reltol = 1e-14, dowarn = FALSE)), error = function(e) { cat("error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(f)) { cat(sprintf("pdynmc nl %-8s", est), format(getb(f), digits = 6), "| se", format(unname(f$stderr), digits = 4), "\n")
    cat("   n moments:", ncol(f$w.mat[[1]]), "\n") }
}
