# Answer conditional-independence queries with RCIT's tests, the R reference for RCoT and KCI.
#
#   Rscript ci_query_worker.R <test> <draws dir> <out.csv>
#
# <draws dir> holds d0.csv, d1.csv, ... with columns x, y, z1..zk; each query tests x against y
# given the z columns. One output row per draw: draw, p, seconds, the time of the test call alone.
# `rcot` is RCIT::RCoT at Andrey's RCoT settings (5 features for x and y, 100 for the conditioning
# set, the Hall-Buckley-Eagleson p-value, seed 0); `kcit` is RCIT::KCIT with its gamma
# approximation (`Bootstrap = FALSE`), as Andrey's KCI.
args <- commandArgs(trailingOnly = TRUE)
test <- args[1]
files <- list.files(args[2], pattern = "^d[0-9]+[.]csv$")
rows <- list()
for (i in seq_along(files) - 1L) {
    data <- as.matrix(read.csv(file.path(args[2], sprintf("d%d.csv", i))))
    x <- data[, 1]
    y <- data[, 2]
    z <- if (ncol(data) > 2) data[, -(1:2), drop = FALSE] else NULL
    start <- proc.time()[["elapsed"]]
    p <- if (test == "rcot") {
        RCIT::RCoT(x, y, z, approx = "hbe", num_f = 100, num_f2 = 5, seed = 0)$p
    } else if (test == "kcit") {
        RCIT::KCIT(x, y, z, Bootstrap = FALSE)
    } else {
        stop("unknown test ", test)
    }
    rows[[i + 1L]] <- data.frame(draw = i, p = p, seconds = proc.time()[["elapsed"]] - start)
}
write.csv(do.call(rbind, rows), args[3], row.names = FALSE)
