###### solution

posterior = function(ppl_function, number_of_iterations) {
  numerator = 0.0
  denominator = 0.0
  for (i in 1:number_of_iterations) {
    weight <<- 1.0
    g_i = ppl_function()
    numerator = numerator + weight * g_i
    denominator = denominator + weight
  }
  return(numerator/denominator)
}

###### contents of the scaffold

suppressPackageStartupMessages(library(distr))

## Utilities to make the distr library a bit nicer to use

p <- function(distribution, realization) {
  d(distribution)(realization) # return the PMF or density 
}

Bern = function(probability_to_get_one) {
  DiscreteDistribution(supp = 0:1, prob = c(1-probability_to_get_one, probability_to_get_one))
}

## Key functions called by simPPLe programs

# Use simulate(distribution) for unobserved random variables
simulate <- function(distribution) {
  r(distribution)(1) # sample once from the given distribution
}

# Use observe(realization, distribution) for observed random variables
observe = function(realization, distribution) {
  # `<<-` lets us modify variables that live in the global scope from inside a function
  weight <<- weight * p(distribution, realization) 
}