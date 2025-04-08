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

# posterior samples
fit = sampling(stan_model(file.choose()), 
               data=list(N=nrow(df), failures=df$numberOfFailures, launches=df$numberOfLaunches),
               iter=5000, control = list(max_treedepth = 15))
samples = extract(fit)$probs
# check Delta 7925H and its failure probabilities
idx = which(df$LV.Type == "Delta 7925H")
samples[idx] # 0.06995836

