import pandas as pd
import numpy as np

shsz = pd.read_csv('.\\data\\csi.csv', thousands=',')
spx = pd.read_csv('.\\data\\spx.csv', thousands=',')
nky = pd.read_csv('.\\data\\nky.csv', thousands=',')
ukx = pd.read_csv('.\\data\\ukx.csv', thousands=',')
asx = pd.read_csv('.\\data\\asx.csv', thousands=',')
bov = pd.read_csv('.\\data\\bov.csv', thousands=',')
cac = pd.read_csv('.\\data\\cac.csv', thousands=',')
dax = pd.read_csv('.\\data\\dax.csv', thousands=',')
hsi = pd.read_csv('.\\data\\hsi.csv', thousands=',')

shsz.Date = shsz.Date.astype('datetime64')
spx.Date = spx.Date.astype('datetime64')
nky.Date = nky.Date.astype('datetime64')
ukx.Date = ukx.Date.astype('datetime64')
asx.Date = asx.Date.astype('datetime64')
bov.Date = bov.Date.astype('datetime64')
cac.Date = cac.Date.astype('datetime64')
dax.Date = dax.Date.astype('datetime64')
hsi.Date = hsi.Date.astype('datetime64')

shsz = shsz.sort_values('Date')
spx = spx.sort_values('Date')
nky = nky.sort_values('Date')
ukx = ukx.sort_values('Date')
asx = asx.sort_values('Date')
bov = bov.sort_values('Date')
cac = cac.sort_values('Date')
dax = dax.sort_values('Date')
hsi = hsi.sort_values('Date')

ptd_shsz = pd.DataFrame({'shsz': shsz.Price.values}, index = shsz.Date.values)
ptd_spx = pd.DataFrame({'spx': spx.Price.values}, index = spx.Date.values)
ptd_nky = pd.DataFrame({'nky': nky.Price.values}, index = nky.Date.values)
ptd_ukx = pd.DataFrame({'ukx': ukx.Price.values}, index = ukx.Date.values)
ptd_asx = pd.DataFrame({'asx': asx.Price.values}, index = asx.Date.values)
ptd_bov = pd.DataFrame({'bov': bov.Price.values}, index = bov.Date.values)
ptd_cac = pd.DataFrame({'cac': cac.Price.values}, index = cac.Date.values)
ptd_dax = pd.DataFrame({'dax': dax.Price.values}, index = dax.Date.values)
ptd_hsi = pd.DataFrame({'hsi': hsi.Price.values}, index = hsi.Date.values)

dta = [ptd_shsz, ptd_spx, ptd_nky, ptd_ukx, ptd_asx, ptd_bov, ptd_cac, ptd_dax, ptd_hsi]
ptd = pd.concat(dta, axis = 1, join = 'outer').iloc[1:, :]
ptd = ptd.fillna(method = 'ffill')
rtd = np.log(ptd) - np.log(ptd.shift(1))
rtd = rtd.iloc[1:,]

rtd.to_csv('.\\data\\rtd.csv')