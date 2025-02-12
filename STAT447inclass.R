suppressPackageStartupMessages(require("dplyr"))
df = read.csv("./hubble-1.csv") %>%
  rename(distance = R..Mpc.) %>%
  rename(velocity = v..km.sec.)
df$velocity = df$velocity/1000
rmarkdown::paged_table(df)
plot(df$distance, df$velocity, xlab = "distance", ylab = "velocity")

source("./simple.R")
source("./simple_utils.R")

regression = function() {
  # priors will be defined here
  slope = simulate(Norm(0,1))
  sd = simulate(Exp(rate=10))
  for (i in 1:nrow(df)) { 
    distance = df[i, "distance"]
    velocity = df[i, "velocity"]
    # likelihood will be defined here
    observe(velocity, Norm(slope * distance, sd))
  }
  return(c(slope, sd))
}

posterior(regression, 1000)


post = posterior_particles(regression, 10000)
weighted_scatter_plot(post, plot_options = list(xlab="slope parameter", ylab="sd parameter"))
