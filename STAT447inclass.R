suppressPackageStartupMessages(require(rstan))
suppressPackageStartupMessages(require(ggplot2))
suppressPackageStartupMessages(require(dplyr))

### hubble example
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



# rocket stan example
set.seed(1)

df = read.csv(url("https://raw.githubusercontent.com/UBC-Stat-ML/web447/main/data/launches.csv")) %>% filter(LV.Type == "Ariane 1")
success_indicators = df$Suc_bin
rmarkdown::paged_table(df)

# data_without_zeros = pmax(pmin(df$percentage/100,0.999),0.001)
rocket = stan_model(file = "D:/rocket.stan")

fit = sampling(
  rocket, 
  data = list(
    y = success_indicators, 
    N = length(success_indicators)
  ), 
  chains = 1,
  iter = 10000      
)

samples = extract(fit)
prob = ifelse(samples$slope > 0, 1, 0)
mean(prob)


