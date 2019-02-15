library(CDVine)

dta <- read.csv('.\\data\\copula_data.csv')
row.names(dta) <- dta[,1]
dta <- dta[,-1]
indices <- c('csi', 'spx', 'nky', 'ukx')

BiCopIndTest(dta$spx, dta$csi)$p.value
BiCopIndTest(dta$spx, dta$nky)$p.value
BiCopIndTest(dta$spx, dta$ukx)$p.value

library(arrangements)

orders <- permutations(c(1, 3, 4))
orders <- cbind(rep(2, 6), orders)
loglik <- 0
family <- rep(2, 6)
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

dta_ <- dta
n <- length(dta_)
stru <- rep(0, n - 1)
indices_ <- indices
for(i in n : 2){
  tau = diag(i)
  for(j in 1 : i){
    for(k in 1 : i){
      est <- BiCopEst(dta_[j, ], dta_[k, ], family = 2)
      tau[j, k] <- abs(BiCopPar2Tau(2, est$par, est$par2))
      est <- BiCopEst(dta_[k, ], dta_[j, ], family = 2)
      tau[k, j] <- abs(BiCopPar2Tau(2, est$par, est$par2))
    }
    tau[j, j] <- 0
  }
  print(tau)
  root <- which.max(colSums(tau))
  stru[n - i + 1] <- indices_[root]
  indices_ <- indices_[-root]
  dta_ <- dta_[, -root]
}

mle_est <- CDVineMLE(dta[, Order], family = family, type = 1)
mle_est <- CDVineMLE(dta[, Order], family = family, type = 1, start = mle_est$par, start2 = mle_est$par2)