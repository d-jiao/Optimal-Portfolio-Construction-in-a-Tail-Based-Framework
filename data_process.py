import pandas as pd
import numpy as np

csi = pd.read_csv('.\\data\\csi.csv', thousands=',')
spx = pd.read_csv('.\\data\\spx.csv', thousands=',')
nky = pd.read_csv('.\\data\\nky.csv', thousands=',')
ukx = pd.read_csv('.\\data\\ukx.csv', thousands=',')
asx = pd.read_csv('.\\data\\asx.csv', thousands=',')
bov = pd.read_csv('.\\data\\bov.csv', thousands=',')
cac = pd.read_csv('.\\data\\cac.csv', thousands=',')
dax = pd.read_csv('.\\data\\dax.csv', thousands=',')
hsi = pd.read_csv('.\\data\\hsi.csv', thousands=',')

csi.Date = csi.Date.astype('datetime64')
spx.Date = spx.Date.astype('datetime64')
nky.Date = nky.Date.astype('datetime64')
ukx.Date = ukx.Date.astype('datetime64')
asx.Date = asx.Date.astype('datetime64')
bov.Date = bov.Date.astype('datetime64')
cac.Date = cac.Date.astype('datetime64')
dax.Date = dax.Date.astype('datetime64')
hsi.Date = hsi.Date.astype('datetime64')

csi = csi.sort_values('Date')
spx = spx.sort_values('Date')
nky = nky.sort_values('Date')
ukx = ukx.sort_values('Date')
asx = asx.sort_values('Date')
bov = bov.sort_values('Date')
cac = cac.sort_values('Date')
dax = dax.sort_values('Date')
hsi = hsi.sort_values('Date')

ptd_csi = pd.DataFrame({'csi': csi.Price.values}, index = csi.Date.values)
ptd_spx = pd.DataFrame({'spx': spx.Price.values}, index = spx.Date.values)
ptd_nky = pd.DataFrame({'nky': nky.Price.values}, index = nky.Date.values)
ptd_ukx = pd.DataFrame({'ukx': ukx.Price.values}, index = ukx.Date.values)
ptd_asx = pd.DataFrame({'asx': asx.Price.values}, index = asx.Date.values)
ptd_bov = pd.DataFrame({'bov': bov.Price.values}, index = bov.Date.values)
ptd_cac = pd.DataFrame({'cac': cac.Price.values}, index = cac.Date.values)
ptd_dax = pd.DataFrame({'dax': dax.Price.values}, index = dax.Date.values)
ptd_hsi = pd.DataFrame({'hsi': hsi.Price.values}, index = hsi.Date.values)

dta = [ptd_csi, ptd_spx, ptd_nky, ptd_ukx, ptd_asx, ptd_bov, ptd_cac, ptd_dax, ptd_hsi]
ptd = pd.concat(dta, axis = 1, join = 'outer').iloc[1:, :]
ptd = ptd.fillna(method = 'ffill')
rtd = np.log(ptd) - np.log(ptd.shift(1))
rtd = rtd.iloc[1:,]

r0 = pd.read_csv('.\\data\\10y.csv', index_col=0)
r0 = pd.DataFrame({'r0': r0.Price.values}, index = pd.to_datetime(r0.index))
rtd = rtd.join(r0 / 365)
rtd = rtd.fillna(method = 'ffill')

ptd.to_csv('.\\data\\ptd.csv')
rtd.to_csv('.\\data\\rtd.csv')