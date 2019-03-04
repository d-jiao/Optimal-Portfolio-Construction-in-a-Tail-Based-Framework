####################################################################

library(CDVine)
library(arrangements)

####################################################################

dta <- read.csv('.\\data\\copula_data.csv')
row.names(dta) <- dta[,1]
dta <- dta[,-1]
indices <- c('csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx')
family <- rep(2, length(dta) * (length(dta) - 1) / 2)

####################################################################

BiCopIndTest(dta$spx, dta$csi)$p.value
BiCopIndTest(dta$spx, dta$nky)$p.value
BiCopIndTest(dta$spx, dta$ukx)$p.value
BiCopIndTest(dta$spx, dta$hsi)$p.value
BiCopIndTest(dta$spx, dta$cac)$p.value
BiCopIndTest(dta$spx, dta$dax)$p.value
BiCopIndTest(dta$spx, dta$asx)$p.value

####################################################################

orders <- permutations(c(1, 3, 4, 5, 6, 7, 8))
orders <- cbind(rep(2, 6), orders)
loglik <- 0
for(i in 1 : dim(orders)[1]){
  order <- orders[i, ]
  dta_ <- dta[, order]
  est <- CDVineMLE(dta_, family = family, type = 1)
  est <- CDVineMLE(dta_, family = family, type = 1, start = est$par, start2 = est$par2)
  if(est$loglik > loglik){
    loglik <- est$loglik
    Order <- order
  }
}

####################################################################

dta_ <- dta[, -2]
n <- length(dta_)
stru <- rep(0, n - 1)
indices_ <- indices[-2]
for(i in n : 2){
  tau = diag(i)
  for(j in 1 : i){
    for(k in 1 : i){
      est <- BiCopEst(dta_[, j], dta_[, k], family = 2)
      tau_ <- abs(BiCopPar2Tau(2, est$par, est$par2))
      tau[j, k] <- tau_
      tau[k, j] <- tau_
    }
    tau[j, j] <- 0
  }
  print(tau)
  root <- which.max(rowSums(tau))
  stru[n - i + 1] <- indices_[root]
  indices_ <- indices_[-root]
  dta_ <- dta_[, -root]
}

stru <- c('spx', stru, indices_)
Order <- match(stru, indices)

####################################################################

mle_est <- CDVineMLE(dta[, Order], family = family, type = 1)
mle_est <- CDVineMLE(dta[, Order], family = family, type = 1, start = mle_est$par, start2 = mle_est$par2)

params <- data.frame(mle_est$par, mle_est$par2, family)
colnames(params) <- c('corr', 'dof', 'family')
write.csv(params, file = paste('.\\data\\cop_param.csv', sep = ''))