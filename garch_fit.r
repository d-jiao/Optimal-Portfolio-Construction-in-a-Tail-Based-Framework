####################################################################

library(rugarch)
library(DistributionUtils)
library(psych)

####################################################################

indices <- c('csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx')
rtd <- read.csv('.\\data\\rtd.csv')
row.names(rtd) <- rtd[,1]
rtd <- rtd[,indices]

####################################################################

summary(rtd)
describe(rtd)

for(i in 1 : length(indices)){
  print(sd(rtd[,indices[i]]))
  print(skewness(rtd[,indices[i]]))
  print(kurtosis(rtd[,indices[i]]))
}

####################################################################

for(i in 1 : length(indices)){
  dta <- data.frame(rtd[,i])
  for(j in c(1, 5, 10)){
    test_res <- Box.test(dta ^ 2, lag = j, type = 'Ljung')
    print(indices[i])
    print(j)
    show(test_res)
  }
}

####################################################################

spec <- ugarchspec(variance.model = list(model = 'gjrGARCH', garchOrder = c(1, 1)), 
                   mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")

time <- as.Date(row.names(rtd))
res <- NULL
vol <- NULL

for(i in 1 : length(indices)){
  dta <- data.frame(rtd[,indices[i]])
  row.names(dta) <- time
  colnames(dta) <- indices[i]
  
  fit <- ugarchfit(spec, data = dta)
  
  res_ <- data.frame(residuals(fit))
  colnames(res_) <- indices[i]
  if(is.null(res)){
    res <- res_
  }else{
    res <- cbind(res, res_)
  }
  
  vol_ <- data.frame(sigma(fit))
  colnames(vol_) <- indices[i]
  if(is.null(vol)){
    vol <- vol_
  }else{
    vol <- cbind(vol, vol_)
  }
  
  show(fit)
}

write.csv(res, file = paste('.\\data\\res.csv', sep = ''))
write.csv(vol, file = paste('.\\data\\vol.csv', sep = ''))
