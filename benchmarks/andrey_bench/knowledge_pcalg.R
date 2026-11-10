# pcalg's PC with edge knowledge, for benchmarks/knowledge_paths.py.
#
#   Rscript knowledge_pcalg.R DATA.csv REQUIRED.csv FORBIDDEN.csv OUT.csv ALPHA
#
# REQUIRED and FORBIDDEN list one directed edge "a,b" (0-based columns) per line. A required edge
# keeps its pair adjacent through the skeleton search (`fixedEdges`). `addBgKnowledge` then
# directs the CPDAG one edge at a time: a required edge a -> b as it is, and a forbidden edge
# a -> b as b -> a when the pair is adjacent and undirected. `fixedGaps` removes an adjacency
# both ways, so it expresses no forbidden direction and is not used. An edge `addBgKnowledge`
# refuses, or a forbidden edge the CPDAG already directs, is counted as unmet and left out.
# OUT gets the adjacency read row -> column, and the last line of standard output the unmet count.
suppressPackageStartupMessages(library(pcalg))
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[1]))
labels <- colnames(X)
d <- ncol(X)
pairs <- function(path) {
    if (file.size(path) == 0) return(matrix(0L, 0, 2))
    as.matrix(read.csv(path, header = FALSE)) + 1L
}
required <- pairs(args[2])
forbidden <- pairs(args[3])
fixed <- matrix(FALSE, d, d)
for (k in seq_len(nrow(required))) {
    fixed[required[k, 1], required[k, 2]] <- fixed[required[k, 2], required[k, 1]] <- TRUE
}
fit <- pc(list(C = cor(X), n = nrow(X)), indepTest = gaussCItest, alpha = as.numeric(args[5]),
          labels = labels, fixedEdges = fixed)
# amat.cpdag writes an edge at the child's row: i -> j is amat[j, i] = 1 and amat[i, j] = 0.
amat <- as(fit, "amat")
amat <- matrix(as.numeric(amat), d, d, dimnames = list(labels, labels))
unmet <- 0L
orient <- function(from, to) {
    out <- tryCatch(
        addBgKnowledge(amat, x = labels[from], y = labels[to], checkInput = FALSE),
        error = function(e) NULL
    )
    if (is.null(out)) unmet <<- unmet + 1L else amat <<- out
}
for (k in seq_len(nrow(required))) invisible(capture.output(orient(required[k, 1], required[k, 2])))
for (k in seq_len(nrow(forbidden))) {
    a <- forbidden[k, 1]
    b <- forbidden[k, 2]
    if (amat[a, b] != 0 && amat[b, a] != 0) {
        invisible(capture.output(orient(b, a)))
    } else if (amat[b, a] != 0) {
        unmet <- unmet + 1L
    }
}
write.table(t(amat), args[4], sep = ",", row.names = FALSE, col.names = FALSE)
cat("unmet", unmet, "\n")
