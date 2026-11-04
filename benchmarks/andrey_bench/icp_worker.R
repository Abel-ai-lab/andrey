# Fit InvariantCausalPrediction::ICP on every draw of a directory, the R reference for andrey.icp.
#
#   Rscript icp_worker.R <draws dir> <alpha> <largest set, "all", or "defaults"> <tests> <fixed>
#
# <draws dir> holds d0.csv, d1.csv, ... with the predictors' columns, then `y`, then `env`; <tests>
# is "normal", "exact", or "normal,exact". Every fit runs at the like-for-like settings of
# benchmarks/comparisons/icp.md: no screening (selection = "all", or "lasso" with every variable
# kept and maxNoVariablesSimult = <largest set> for a size cap), no subsampling of the exact test
# (maxNoObs above every environment's rows), and the full search (stopIfEmpty = FALSE).
# "defaults" runs ICP() at the package's own defaults instead, but for alpha and test: screening by
# boosting above 8 variables, sets of at most 8, and subsamples of 200 rows in the exact test.
#
# <draws dir>/fits.tsv gets one row per draw and test: the seconds of the ICP() call alone, the
# p-value of the empty set (ICP() leaves the empty set out of acceptedSets under "lasso"), and
# ICP()'s modelReject, bestModel, per-variable pvalues, and ConfInt. Its acceptedSets, up to every
# subset of the variables, go to <draws dir>/accepted_<draw>_<test>.txt, one set per line: its
# 1-based predictor positions joined by ",", "{}" for the empty set.
# With <fixed> = 1, <draws dir>/sets_<draw>.tsv also gets every tested set's p-value from the
# package's own getpval under both tests, with Chow's p-value computed here from the package's
# residual variance and from the OLS residual variance, and <draws dir>/coef_<draw>_<test>.tsv
# the coefficients and standard errors of each accepted set, from Coeff and CoeffVar.
suppressPackageStartupMessages(library(InvariantCausalPrediction))
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
alpha <- as.numeric(args[2])
largest <- args[3]
tests <- strsplit(args[4], ",")[[1]]
fixed <- args[5] == "1"

join <- function(v) paste(format(v, digits = 17, scientific = TRUE, trim = TRUE), collapse = ";")
set_text <- function(s) if (length(s) == 0) "{}" else paste(s, collapse = ",")

# Chow's test of every environment against the others, eq. 17 of Peters et al. (2016), with the
# package's residual variance (getpval.R line 65) or the OLS one, RSS / (n_out - p).
chow <- function(Y, X, IN, sigma) {
    n <- nrow(X)
    p_env <- sapply(IN, function(I) {
        fit <- lm.fit(X[-I, , drop = FALSE], Y[-I])
        r <- fit$residuals
        D <- Y[I] - X[I, , drop = FALSE] %*% fit$coefficients
        S <- diag(length(I)) + X[I, , drop = FALSE] %*%
            solve(crossprod(X[-I, , drop = FALSE]), t(X[I, , drop = FALSE]))
        n_out <- n - length(I)
        s2 <- if (sigma == "package") var(r) * n_out / (n_out - ncol(X)) else sum(r^2) / (n_out - ncol(X))
        stat <- drop(t(D) %*% solve(S, D)) / (length(I) * s2)
        pf(stat, length(I), n_out - ncol(X), lower.tail = FALSE)
    })
    min(1, length(IN) * min(p_env))
}

files <- list.files(dir, pattern = "^d[0-9]+[.]csv$")
rows <- list()
for (i in seq_along(files) - 1L) {
    data <- as.matrix(read.csv(file.path(dir, sprintf("d%d.csv", i))))
    p <- ncol(data) - 2L
    X <- data[, seq_len(p), drop = FALSE]
    Y <- data[, p + 1L]
    E <- data[, p + 2L]
    IN <- lapply(sort(unique(E)), function(k) which(E == k))
    selection <- if (largest == "all") "all" else "lasso"
    simult <- if (largest %in% c("all", "defaults")) p else as.integer(largest)
    for (test in tests) {
        start <- proc.time()[["elapsed"]]
        f <- if (largest == "defaults") {
            ICP(X, Y, E, alpha = alpha, test = test, showAcceptedSets = FALSE,
                showCompletion = FALSE)
        } else {
            ICP(X, Y, E, alpha = alpha, test = test, selection = selection,
                maxNoVariables = p, maxNoVariablesSimult = simult,
                maxNoObs = .Machine$integer.max, stopIfEmpty = FALSE,
                showAcceptedSets = FALSE, showCompletion = FALSE)
        }
        seconds <- proc.time()[["elapsed"]] - start
        empty <- InvariantCausalPrediction:::getpval(Y, matrix(1, nrow(X), 1), IN, test = test,
                                                     maxNoObs = .Machine$integer.max)$pval
        conf <- if (is.null(f$ConfInt)) matrix(NA, 2, p) else f$ConfInt
        rows[[length(rows) + 1L]] <- data.frame(
            draw = i, test = test, seconds = seconds, empty_p = format(empty, digits = 17),
            model_reject = f$modelReject, best = format(f$bestModel, digits = 17),
            pvalues = join(f$pvalues), upper = join(conf[1, ]), lower = join(conf[2, ])
        )
        writeLines(vapply(f$acceptedSets, set_text, ""),
                   file.path(dir, sprintf("accepted_%d_%s.txt", i, test)))
        if (fixed) {
            used <- rep(0L, p)
            sets <- character(0); coef <- character(0); se <- character(0)
            for (s in f$acceptedSets) {
                if (length(s) == 0) next
                used[s] <- used[s] + 1L
                sets <- c(sets, set_text(s))
                coef <- c(coef, join(mapply(function(k, u) f$Coeff[[k]][u], s, used[s])))
                se <- c(se, join(mapply(function(k, u) f$CoeffVar[[k]][u], s, used[s])))
            }
            write.table(data.frame(set = sets, coef = coef, se = se),
                        file.path(dir, sprintf("coef_%d_%s.tsv", i, test)),
                        sep = "\t", row.names = FALSE, quote = FALSE)
        }
    }
    if (fixed) {
        size <- if (largest == "all") p else as.integer(largest)
        sets <- unlist(lapply(0:size, function(k) combn(seq_len(p), k, simplify = FALSE)),
                       recursive = FALSE)
        if (p == 0) sets <- list(integer(0))
        table <- do.call(rbind, lapply(sets, function(s) {
            Xs <- cbind(1, X[, s, drop = FALSE])
            data.frame(
                set = set_text(s),
                normal = format(InvariantCausalPrediction:::getpval(Y, Xs, IN, test = "normal")$pval,
                                digits = 17),
                exact = format(InvariantCausalPrediction:::getpval(Y, Xs, IN, test = "exact",
                                                                   maxNoObs = .Machine$integer.max)$pval,
                               digits = 17),
                chow_package = format(chow(Y, Xs, IN, "package"), digits = 17),
                chow_paper = format(chow(Y, Xs, IN, "paper"), digits = 17)
            )
        }))
        write.table(table, file.path(dir, sprintf("sets_%d.tsv", i)), sep = "\t",
                    row.names = FALSE, quote = FALSE)
    }
}
write.table(do.call(rbind, rows), file.path(dir, "fits.tsv"), sep = "\t", row.names = FALSE,
            quote = FALSE)
