suppressPackageStartupMessages(require(rstan))
suppressPackageStartupMessages(require(ggplot2))
suppressPackageStartupMessages(require(dplyr))
set.seed(2025)


# data from: https://data.chhs.ca.gov/dataset/covid-19-variant-data
df = read.csv(file.choose())
df$date = as.Date(df$date,format="%Y-%m-%d")

df = df %>% filter(date > "2021-12-01" & date < "2021-12-31") %>% filter(variant_name == "Omicron")

df %>% ggplot(aes(x = date, y = percentage)) + geom_point() + ylim(0, 100) + theme_minimal()

data_without_zeros = pmax(pmin(df$percentage/100,0.999),0.001)

fit = sampling(
  stan_model(file.choose()), 
  data = list(
    y = data_without_zeros, 
    N = length(data_without_zeros)
  ), 
  chains = 1,
  iter = 10000       
)

samples = extract(fit)
# N = nrow(samples$mu)
# samples_mu = matrix(, nrow=nrow(samples$mu), ncol=ncol(samples$mu))
# for (i in 1:N) {
#   samples_mu[i,] = plogis(samples$theta1[i]*seq(1,ncol(samples$mu),1)/ncol(samples$mu) + samples$theta2[i])
# }
samples_mu = samples$mu
data = df$percentage
n_samples = nrow(samples_mu)
xs = 1:length(data) / length(data)
plot(xs, data/100, 
     xlab = "Fraction of the month of December, 2021", 
     ylab = "Omicron fraction")

for (i in 1:n_samples) {
  lines(xs, samples_mu[i,], col = rgb(red = 0, green = 0, blue = 0, alpha = 0.01))
}

