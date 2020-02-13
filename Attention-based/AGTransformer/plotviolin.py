import pandas as pd

import xlrd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
data = xlrd.open_workbook('data_motion.xlsx')

# print 'sheet_names:', data.sheet_names()
# print 'sheet_number:', data.nsheets
per = data.sheet_by_name("Personality")

# print "sheet name:", per.name   # get sheet name
# print "per row num:", per.nrows  # get sheet all rows number
# print "per col num:", per.ncols

personality = pd.read_excel('data_motion.xlsx', sheet_name = "Personality")
# print type(personality)

# print(personality)
# print(personality.iloc[:,1])
scale = ['Extraversion', 'Agreeableness', 'Conscientiusness', 'Emotional stability', 'Creativity']
newPfive=pd.DataFrame(columns=scale)


for i in range(0,5):
	if (i+1)%2!=0:
		newPfive[scale[i]] = (personality.iloc[:,i+1] + (8-personality.iloc[:,i+1+5]))/2.0
	else:
		newPfive[scale[i]] = (8-personality.iloc[:,i+1] + personality.iloc[:,i+1+5])/2.0

# print personality.head(5)
print(newPfive['Extraversion'].describe())
# fig, axes = plt.subplots(5,1)
ax = sns.violinplot(data=newPfive,inner='quartile', linewidth=3, palette='muted')
# ax = sns.violinplot(y = personality['extraversion'], ax = axes[0])
# ax = sns.violinplot(y = personality['agreeableness'], ax = axes[1])
# ax = sns.violinplot(y = personality['conscientiusness'], ax = axes[2])
# ax = sns.violinplot(y = personality['emotionalstability'], ax = axes[3])
# ax = sns.violinplot(y = personality['opennesstoexperiences'], ax = axes[4])
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(25)
plt.show()