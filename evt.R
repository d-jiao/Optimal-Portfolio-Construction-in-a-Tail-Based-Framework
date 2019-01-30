library(POT)

res <- read.csv('.\\data\\res.csv')
row.names(res) <- res[,1]
res <- res[,-1]

indices <- c('csi', 'spx', 'nky', 'ukx')

summary(res)

time <- as.numeric(as.Date(row.names(res)))
for(i in 1 : length(indices)){
  obs <- res[, i]
  dta <- data.frame(time, obs)
  
  a <- mrlplot(dta[, "obs"])
  b <- diplot(dta)
  c <- tcplot(dta[, "obs"])
  
  mrl_di <- data.frame(a$x, a$y, b$thresh, b$DI)
  write.csv(mrl_di, file = paste('.\\data\\', indices[i], '_mrl_di.csv', sep = ''))
  
  tc <- data.frame(c$scales, c$shapes)
  write.csv(tc, file = paste('.\\data\\', indices[i], '_tc.csv', sep = ''))
}

upper_threshold <- c(0.03, 0.025, 0.03, 0.02)
ur <- c()

for(i in 1 : length(upper_threshold)){
  ur[i] <- sum(res[, i] < upper_threshold[i]) / length(res[, i])
}


# npy <- sum(shsz$obs > 0.025) / 10
# mle <- fitgpd(shsz[, "obs"], thresh = 0.025, est = "mle")
# par(mfrow = c(2, 2))
# plot(mle, npy = npy)

for(i in 1 : length(indices)){
  obs <- -res[, i]
  dta <- data.frame(time, obs)
  
  a <- mrlplot(dta[, "obs"])
  b <- diplot(dta)
  c <- tcplot(dta[, "obs"])
  
  mrl_di <- data.frame(a$x, a$y, b$thresh, b$DI)
  write.csv(mrl_di, file = paste('.\\data\\', indices[i], '_mrl_di_left.csv', sep = ''))
  
  tc <- data.frame(c$scales, c$shapes)
  write.csv(tc, file = paste('.\\data\\', indices[i], '_tc_left.csv', sep = ''))
}

lower_threshold <- c(-0.035, -0.025, -0.03, -0.02)
ul <- c()

for(i in 1 : length(lower_threshold)){
  ul[i] <- sum(res[, i] < lower_threshold[i]) / length(res[, i])
}