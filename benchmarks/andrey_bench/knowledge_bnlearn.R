# bnlearn's searches with edge knowledge, for benchmarks/knowledge_paths.py.
#
#   Rscript knowledge_bnlearn.R ALGO DATA.csv REQUIRED.csv FORBIDDEN.csv OUT.csv ALPHA
#
# ALGO is hc, tabu, mmhc, rsmax2, iamb, or inter_iamb. REQUIRED and FORBIDDEN list one directed
# edge "a,b" (0-based columns) per line: a required edge is a `whitelist` arc and a forbidden one a
# `blacklist` arc. The score is `bic-g` and the Gaussian test Fisher's z (`zf`) at ALPHA. A search
# that returns a DAG (hc, tabu, mmhc, rsmax2) is written as its CPDAG, the arcs the knowledge
# fixes kept directed (`cpdag(wlbl = TRUE)`). OUT gets the adjacency read row -> column, an
# undirected edge in both entries.
suppressPackageStartupMessages(library(bnlearn))
args <- commandArgs(trailingOnly = TRUE)
algo <- args[1]
X <- read.csv(args[2])
labels <- colnames(X)
alpha <- as.numeric(args[6])
arcs <- function(path) {
    if (file.size(path) == 0) return(NULL)
    m <- as.matrix(read.csv(path, header = FALSE)) + 1L
    data.frame(from = labels[m[, 1]], to = labels[m[, 2]])
}
wl <- arcs(args[3])
bl <- arcs(args[4])
fit <- switch(algo,
    hc = hc(X, whitelist = wl, blacklist = bl, score = "bic-g"),
    tabu = tabu(X, whitelist = wl, blacklist = bl, score = "bic-g"),
    mmhc = mmhc(X, whitelist = wl, blacklist = bl, restrict.args = list(test = "zf", alpha = alpha),
                maximize.args = list(score = "bic-g")),
    rsmax2 = rsmax2(X, whitelist = wl, blacklist = bl, restrict = "mmpc", maximize = "hc",
                    restrict.args = list(test = "zf", alpha = alpha),
                    maximize.args = list(score = "bic-g")),
    iamb = iamb(X, whitelist = wl, blacklist = bl, test = "zf", alpha = alpha),
    inter_iamb = inter.iamb(X, whitelist = wl, blacklist = bl, test = "zf", alpha = alpha),
    stop(paste("unknown algorithm", algo))
)
if (algo %in% c("hc", "tabu", "mmhc", "rsmax2")) fit <- cpdag(fit, wlbl = TRUE)
write.table(amat(fit), args[5], sep = ",", row.names = FALSE, col.names = FALSE)
