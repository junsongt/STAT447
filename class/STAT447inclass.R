suppressPackageStartupMessages(require(rstan))
suppressPackageStartupMessages(require(ggplot2))
suppressPackageStartupMessages(require(dplyr))
suppressPackageStartupMessages(require(bayesplot))
set.seed(2025)

### hubble example (simPPLe application in regression)
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



# rocket Stan example
df = read.csv(url("https://raw.githubusercontent.com/UBC-Stat-ML/web447/main/data/launches.csv")) %>% filter(LV.Type == "Ariane 1")
success_indicators = df$Suc_bin
rmarkdown::paged_table(df)

# data_without_zeros = pmax(pmin(df$percentage/100,0.999),0.001)
fit = sampling(
  stan_model(file.choose()), 
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



# Kumaraswamy(user defined Stan distribution) example
fit = sampling(
  stan_model(file.choose()),
  seed = 1,
  refresh = 0,
  data = list(a = 2, b = 5),       
  iter = 10000                   
)

samples = extract(fit)$x
mcmc_hist(fit, pars = c("x")) + theme_minimal()
hist(samples)


# Chernobyl example (censoring)
# detection limit: value higher than that stay at the limit
limit = 1.1 
n_measurements = 10
true_mean = 5.6

# data, if we were able to observe perfectly
y = rexp(n_measurements, 1.0/true_mean)

# number of measurements higher than the detection limit
n_above_limit = sum(y >= limit)
n_below_limit = sum(y < limit)

# subset of the measurements that are below the limit:
data_below_limit = y[y < limit]

# measurements: those higher than the limit stay at the limit
measurements = ifelse(y < limit, y, limit)


fit = sampling(
  stan_model(file.choose()),
  data = list(
    N = n_measurements,
    L = limit,
    n = n_below_limit,
    observed = data_below_limit),
  iter = 10000,
  chain = 1,
  control = list(max_treedepth = 15)
  )
               
# fit = sampling(
#   stan_model(file.choose()),
#   chains = 1,
#   data = list(
#     limit = limit,
#     n_above_limit = n_above_limit, 
#     n_below_limit = n_below_limit,
#     data_below_limit = data_below_limit
#   ),       
#   iter = 100000                   
# )

samples = extract(fit)$x
mean(samples)
mcmc_hist(fit, pars = c("x")) + theme_minimal()

mcmc_areas_ridges(fit, pars = c("mean")) + 
  theme_minimal() + 
  scale_x_continuous(limits = c(0, 10)) 
