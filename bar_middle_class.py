import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import matplotlib as mpl
# plt.style.use('ggplot')
# del mpl.font_manager.weight_dict['roman']
# mpl.font_manager._rebuild()
# params={'font.family':'sans-serif',
#         'font.sans-serif':'Calibri',
#         'font.weight':'normal',
#         }
# mpl.rcParams.update(params)

data = pd.read_csv('.\\data\\middle-class forecast.csv', index_col = 0)
n_groups = data.columns.__len__()

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots(figsize = (8, 4))

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.5
error_config = {'ecolor': '0.3'}

for i in range(len(data)):
    rects1 = ax.bar(index + bar_width * (i - 1), data.iloc[i,:], bar_width,
                alpha=opacity,
                error_kw=error_config,
                label=data.index[i])

ax.set_xlabel('Region')
ax.set_ylabel('Population in Millions')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Asia-Pacific', 'Europe', 'North America ',
       'Central and \nSouth America ', 'Middle East and \nNorth Africa',
       'Sub-Saharan \nAfrica'])
ax.yaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
ax.legend()

fig.tight_layout()
plt.show()
fig.savefig('.\\fig\\middle_class.png')