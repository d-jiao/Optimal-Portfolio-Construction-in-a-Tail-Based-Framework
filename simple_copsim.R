####################################################################

library("CDVine")

####################################################################

u1 <- seq(0.02, 0.98, by = 0.02)
u2 <- seq(0.02, 0.98, by = 0.02)
copPDF1 <- matrix(0, 49, 49)
copPDF2 <- matrix(0, 49, 49)
copPDF3 <- matrix(0, 49, 49)
for(i in 1:49){
  for(j in 1:49){
    copPDF1[i, j] <- BiCopPDF(u1 = u1[i], u2 = u2[j], family = 2, par = 0.9, par2 = 9)
    copPDF2[i, j] <- BiCopPDF(u1 = u1[i], u2 = u2[j], family = 2, par = 0.9, par2 = 3)
    copPDF3[i, j] <- BiCopPDF(u1 = u1[i], u2 = u2[j], family = 2, par = 0.3, par2 = 3)
  }
}  

write.csv(copPDF1, file  =  '.\\data\\coppdf1.csv')
write.csv(copPDF2, file  =  '.\\data\\coppdf2.csv')
write.csv(copPDF3, file  =  '.\\data\\coppdf3.csv')

####################################################################

set.seed(10)
dat1 <- BiCopSim(N = 500, family = 2, par = 0.9, par2 = 9)
dat2 <- BiCopSim(N = 500, family = 2, par = 0.9, par2 = 3)
dat3 <- BiCopSim(N = 500, family = 2, par = 0.3, par2 = 3)

write.csv(dat1, file  =  '.\\data\\copdat1.csv')
write.csv(dat2, file  =  '.\\data\\copdat2.csv')
write.csv(dat3, file  =  '.\\data\\copdat3.csv')