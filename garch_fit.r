library(rugarch)

rtd <- read.csv('.\\data\\rtd.csv')
row.names(rtd) <- rtd[,1]
rtd <- rtd[,c(2, 3, 4, 5)]

indices <- c('csi', 'spx', 'nky', 'ukx')

summary(rtd)

spec <- ugarchspec(variance.model = list(model = 'gjrGARCH', garchOrder = c(1, 1)), 
                   mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")

time <- as.Date(row.names(rtd))
res <- NULL
vol <- NULL

for(i in 1 : length(indices)){
  dta <- data.frame(rtd[,1])
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
