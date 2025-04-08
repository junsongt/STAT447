set.seed(2025)
suppressPackageStartupMessages(require(rstan))
suppressPackageStartupMessages(require(ggplot2))
suppressPackageStartupMessages(require(dplyr))

df = read.csv(url("https://raw.githubusercontent.com/UBC-Stat-ML/web447/main/data/launches.csv")) %>%
  select(LV.Type, Suc)  %>%
  group_by(LV.Type) %>%
  summarise(
    numberOfLaunches = n(),
    numberOfFailures = sum(Suc == "F")
  )
ggplot(df, aes(x = numberOfFailures / numberOfLaunches)) +
  geom_histogram() + 
  xlab("pi_hat = numberOfFailures_i / numberOfLaunches_i") +
  geom_rug(alpha = 0.1) + 
  theme_minimal()



fit = sampling(stan_model(file.choose()), 
               data=list(N=nrow(df), failures=df$numberOfFailures, launches=df$numberOfLaunches),
               iter=5000, control = list(max_treedepth = 15))
samples = extract(fit)$probs
xs = df$LV.Type
n_samples = nrow(samples)
ys = df$numberOfFailures / df$numberOfLaunches


# check some specific type of rocket and their failure probabilities
# choose first 20 rockets
subset = c(1:20)
plot(subset, ys[subset],
     xlab = paste0("Types of rocket: ", xs[subset]), 
     ylab = "probability of failure")

for (i in 1:n_samples) {
  lines(subset, samples[i,subset], col = rgb(red = 0, green = 0, blue = 0, alpha = 0.01))
}

