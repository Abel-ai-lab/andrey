# Show the four places where InvariantCausalPrediction 0.8 departs from Peters, Buhlmann and
# Meinshausen (2016), each on a small seeded dataset; benchmarks/comparisons/icp.md cites them.
#
#   Rscript icp_probes.R
#
# 1. The approximate test multiplies the smallest environment's p-value by K - 1 (getpval.R line
#    42), where the paper's Bonferroni correction is over the K environments.
# 2. stopIfEmpty stops only on an accepted empty set: ICP.R line 158 tests length(intersect), the
#    base function, not the intersection of the accepted sets.
# 3. The exact test's residual variance is var(r) n_out / (n_out - p) (getpval.R line 65), larger
#    than the OLS residual variance RSS / (n_out - p) by n_out / (n_out - 1).
# 4. With more rows in an environment than maxNoObs, the exact test's subsample pairs each
#    prediction error with another row's covariance (getpval.R lines 60-63).
suppressPackageStartupMessages(library(InvariantCausalPrediction))
getpval <- InvariantCausalPrediction:::getpval

cat("1. The approximate test's correction, the empty set in three environments\n")
set.seed(1)
n <- 600
E <- rep(1:3, each = n / 3)
X1 <- rnorm(n) * ifelse(E == 3, 1.3, 1)
Y <- 0.7 * X1 + rnorm(n)
IN <- lapply(1:3, function(k) which(E == k))
r <- Y - mean(Y)
p_env <- sapply(IN, function(i) 2 * min(t.test(r[i], r[-i])$p.value, var.test(r[i], r[-i])$p.value))
cat(sprintf("   getpval %.12g, (K - 1) * min %.12g, K * min %.12g\n",
            getpval(Y, matrix(1, n, 1), IN, test = "normal")$pval, 2 * min(p_env), 3 * min(p_env)))

cat("2. stopIfEmpty with disjoint accepted sets\n")
set.seed(2)
E <- rep(1:2, each = 300)
Z <- rnorm(600) * ifelse(E == 2, 3, 1)
X <- cbind(x1 = Z + 0.05 * rnorm(600), x2 = Z + 0.05 * rnorm(600), x3 = rnorm(600))
Y <- Z + rnorm(600)
for (stop in c(FALSE, TRUE)) {
    f <- ICP(X, Y, E, alpha = 0.05, selection = "all", stopIfEmpty = stop,
             showAcceptedSets = FALSE, showCompletion = FALSE)
    sets <- vapply(f$acceptedSets, function(s) paste0("{", paste(s, collapse = ","), "}"), "")
    cat(sprintf("   stopIfEmpty = %s: accepted %s\n", stop, paste(sets, collapse = " ")))
}

cat("3. The exact test's residual variance, 120 and 80 rows\n")
set.seed(3)
E <- rep(1:2, c(120, 80))
IN <- list(which(E == 1), which(E == 2))
X1 <- rnorm(200) * ifelse(E == 2, 1.7, 1)
Y <- 0.6 * X1 + rnorm(200)
chow <- function(Y, X, IN, sigma) {
    p_env <- sapply(IN, function(I) {
        fit <- lm.fit(X[-I, , drop = FALSE], Y[-I])
        res <- fit$residuals
        D <- Y[I] - X[I, , drop = FALSE] %*% fit$coefficients
        S <- diag(length(I)) + X[I, , drop = FALSE] %*%
            solve(crossprod(X[-I, , drop = FALSE]), t(X[I, , drop = FALSE]))
        n_out <- nrow(X) - length(I)
        s2 <- if (sigma == "package") var(res) * n_out / (n_out - ncol(X)) else sum(res^2) / (n_out - ncol(X))
        pf(drop(t(D) %*% solve(S, D)) / (length(I) * s2), length(I), n_out - ncol(X),
           lower.tail = FALSE)
    })
    min(1, length(IN) * min(p_env))
}
for (X in list(matrix(1, 200, 1), cbind(1, X1))) {
    cat(sprintf("   getpval %.12g, with the package's variance %.12g, with RSS / (n_out - p) %.12g\n",
                getpval(Y, X, IN, test = "exact", maxNoObs = 10^6)$pval,
                chow(Y, X, IN, "package"), chow(Y, X, IN, "paper")))
}

cat("4. The exact test's subsample, 200 rows per environment, maxNoObs = 50\n")
set.seed(5)
E <- rep(1:2, each = 200)
IN <- list(which(E == 1), which(E == 2))
X1 <- rnorm(400) * ifelse(E == 2, 2, 1)
Y <- 0.5 * X1 + rnorm(400)
X <- cbind(1, X1 + 0.6 * rnorm(400))
for (seed in c(7, 8)) {
    set.seed(seed)
    package <- getpval(Y, X, IN, test = "exact", maxNoObs = 50)$pval
    set.seed(seed)  # the same draws, each difference next to its own row
    p_env <- sapply(IN, function(I) {
        fit <- lm.fit(X[-I, ], Y[-I])
        n_out <- 400 - length(I)
        sel <- sample(I, 50)
        D <- Y[sel] - X[sel, ] %*% coefficients(fit)
        S <- diag(50) + X[sel, ] %*% solve(crossprod(X[-I, ]), t(X[sel, ]))
        stat <- (t(D) %*% solve(S, D)) / (50 * var(residuals(fit)) * n_out / (n_out - ncol(X)))
        1 - pf(stat, 50, 400 - 50 - ncol(X))
    })
    cat(sprintf("   seed %d: package %.6f, rows matched %.6f\n", seed, package, min(1, 2 * min(p_env))))
}
