
#devtools::install_github("SWotherspoon/QPress")

library(dplyr)
library(QPress)
library(ggplot2)

format_names <- function(z) tolower(gsub("[ /\\-]+","_",z))

# Read in input data
y <- read.csv("Phillip_Island_input_qualitative_models.csv")

## drop minor interactions and reformat names
y <- y %>% filter(importance!="Minor") %>%
  mutate(to=format_names(to),from=format_names(from))
y

## check node names are consistent
sort(unique(c(y$from,y$to)))

##Convert data to text format suitable to be parsed by the `QPress` package:
model_text <- c()
for (ri in 1:nrow(y)) {
  model_text <- c(model_text,switch(tolower(y$type[ri]),
                                    negative=paste0(y$from[ri],"-*",y$to[ri]),
                                    positive=paste0(y$from[ri],"->",y$to[ri]),
                                    "predator-prey"=paste0(y$from[ri],"<-*",y$to[ri]),
                                    stop("unknown type: ",y$type[ri])))
}
model <- parse.digraph(model_text)

#Check that we don't have duplicate edges
model %>% group_by(From,To) %>% filter(n()>1) %>% select(From,To,Type) %>% arrange(From,To)

## drop duplicate edges from model_text
## these appear as e.g. prey->pred as well as pred<-*prey, so only need one
model_text <- model_text[!model_text %in% c("european_rabbit->red_fox")]

## re-parse
model <- parse.digraph(model_text)

## and recheck
model %>% group_by(From,To) %>% mutate(duplicated=n()>1) %>%
  ungroup %>% filter(duplicated) %>% arrange(From,To)


# Qualitative simulations
# Construct a function to test a validation criterion
validate <- press.validate(model, perturb = c(red_fox = -1), monitor = c(little_penguin = 1, cape_barren_geese = 1))

#Enforce self-limitations on all nodes:
model <- enforce.limitation(model)

#Draw 10000 random sets of interaction weights:
sim <- system.simulate(10000, model, validators = list(validate))

## sim represents a suite of models that meet our validation critera, 
## so we can now ask those models what they predict about the response of other nodes
p1 <- impact.barplot0(sim, perturb = c(red_fox = -1),
                      monitor = NA)
print(p1)


## Foxes are eradicated from our system so we now remove this node
## Create drop nodes function
drop.nodes <- function(sim, to.drop, method = "remove") {
    method <- match.arg(tolower(method), c("remove", "zeros"))
    if (!all(to.drop %in% node.labels(sim$edges))) stop("not all of the to.drop nodes appear in this system (check `node.labels(sim$edges)`)")
    ## indexes of to.drop nodes
    vidx <- sim$edges$From == to.drop | sim$edges$To == to.drop
    vidx2 <- attr(sim$edges, "node.labels") == to.drop
    ## check that each row in w still represents a stable system
    n.nodes <- length(node.labels(sim$edges))
    k.edges <- as.vector(unclass(sim$edges$To)+(unclass(sim$edges$From)-1)*n.nodes)
    ## each row in w corresponds to W[k.edges] where W is a full (square, sparse) weights matrix
    ## we need W to test stability
    sim_new <- sim
    if (method == "remove") {
        sim_new$edges <- sim_new$edges[!vidx, ]
        sim_new$edges$From <- droplevels(sim_new$edges$From)
        sim_new$edges$To <- droplevels(sim_new$edges$To)
        ## build the same k.edge indexer for the reduced edge set
        k.edges_new <- as.vector(unclass(sim_new$edges$To)+(unclass(sim_new$edges$From)-1)*length(node.labels(sim_new$edges)))
        w_new <- matrix(NA_real_, nrow = nrow(sim$w), ncol = sum(!vidx))
    } else {
        k.edges_new <- k.edges
        w_new <- matrix(NA_real_, nrow = nrow(sim$w), ncol = ncol(sim$w))
    }
    A_new <- list()
    accepted <- 0L
    for (wi in seq_len(nrow(sim$w))) {
        Wnew <- matrix(0, nrow = n.nodes, ncol = n.nodes)
        Wnew[k.edges] <- sim$w[wi, ]
        if (method == "remove") {
            Wnew <- Wnew[!vidx2, !vidx2] ## drop edges connected to our unwanted node(s)
            ## double-check that this indexing is correct:
            stopifnot(all(Wnew[k.edges_new] == sim$w[wi, !vidx]))
        } else {
            ## set edge weights to zero
            self <- diag(Wnew)
            Wnew[vidx2, ] <- 0
            Wnew[, vidx2] <- 0
            diag(Wnew) <- self ## reinstate all self-lims, including on "removed" nodes
        }
        ## Wnew has to be stable
        if (stable.community(Wnew)) {
            accepted <- accepted + 1L
            w_new[accepted, ] <- Wnew[k.edges_new]
            A_new[[accepted]] <- -solve(Wnew)
        }
    }
    sim_new$A <- A_new
    sim_new$w <- w_new[seq_len(accepted), ]
    sim_new$accepted <- accepted
    sim_new
}

## Remove fox nodes
sim_no_foxes <- drop.nodes(sim, "red_fox")

## Assess run to run variability in results
Nruns <- 100
rrv <- lapply(seq_len(Nruns), function(z) {
  sim <- system.simulate(10000, model, validators = list(validate))
  sim_no_foxes <- drop.nodes(sim, "red_fox")
  pnf <- impact.barplot0(sim_no_foxes, perturb = c(feral_cat = -1))
  pnf
})

## convert to fraction of simulations, because the total number won't be identical for each rep
rrvn <- array(NA_real_, dim = c(dim(rrv[[1]])[1:2], length(rrv)))
for (i in seq_along(rrv)) rrvn[, , i] <- rrv[[i]]/max(rrv[[i]]) 

rr <- apply(rrvn, c(1, 2), function(z) diff(range(z)))
max(rr) ## maximum difference in result across the 100 runs
mean(rr[rr > 0]) ## mean difference in result across the 100 runs

## visual representation of parameter ranges
wdf <- bind_rows(lapply(seq_len(ncol(sim$w)), 
                        function(z) data.frame(w = as.numeric(sim$w[, z]), 
                                               edge = colnames(sim$w)[z])))
ggplot(wdf, aes(w, after_stat(ncount))) + 
  geom_histogram(breaks = seq(-1, 1, length.out = 30), closed = "left") + 
  facet_wrap(~edge) + theme_bw()

## explore what happens when we suppress cats
p_no_foxes <- impact.barplot0(sim_no_foxes, perturb = c(feral_cat = -1))
p_no_foxes

## explore what happens when we suppress rabbits
OC_no_foxes <- impact.barplot0(sim_no_foxes, perturb = c(european_rabbit = -1))
OC_no_foxes

## explore what happens when we suppress rodents
Muridae_no_foxes <- impact.barplot0(sim_no_foxes, perturb = c(black_rat = -1, house_mouse = -1))
Muridae_no_foxes

