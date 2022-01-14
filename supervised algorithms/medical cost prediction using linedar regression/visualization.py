sns.set_style("whitegrid")
data.plot.box()                         #this line visualize outliers in the data


fig, ax = plt.subplots(figsize = (10,6))                #this block of code is another way to visualize outliers in the data
ax.scatter(data['bmi'],data['charges'],color = 'red')
ax.set_xlabel("bmi")
ax.set_ylabel("charges")
plt.show()



sns.boxplot(x = data['bmi'])        #this code visualize the feature column 'bmi'


sns.boxplot(x = data['charges'])      #this does the same but with the charges column


charges = data['charges'].groupby(data.region).sum().sort_values(ascending = True)      # this code is to see which regions has the highest and lowest charge
f , ax = plt.subplots(1,1,figsize = (10,8))
ax = sns.barplot(charges.head(), charges.head().index, palette = 'Greens')




f , ax = plt.subplots(1,1, figsize = (12,8))          # this block of code shows how charges are in different regions but with smokers
ax = sns.barplot(x = 'region' , y = 'charges', hue = 'smoker', data = data , palette = 'Blues')



f , ax = plt.subplots(1,1,figsize = (15,12))            #Now with childrens
ax = sns.barplot(x = 'region', y = 'charges', hue = 'children', data =data, palette = 'Set1')



ax = sns.lmplot(x = 'age', y = 'charges' , hue = 'smoker', data = data, palette = 'Set1')         #Now let's analyze the medical charges by age, bmi(body mass index) and 
ax = sns.lmplot(x = 'bmi', y = 'charges' , hue = 'smoker', data = data, palette = 'Set2')         #children according to the smoking factor
ax = sns.lmplot(x = 'children', y = 'charges' , hue = 'smoker', data = data, palette = 'Set3')



f,ax=plt.subplots(1,1,figsize=(10,10))                                                #this is the violin plot
ax=sns.violinplot(x='children',y='charges',data=data,orient='v',hue='smoker',palette='inferno')




sns.pairplot(data)          #Now look at the whole data





proff = ProfileReport(data)           #Now look at the Profile Report of this Dataset



data['region'].value_counts()           #counts regions values



sns.countplot(x = 'region', data = data)        #countplot visualization



sns.countplot(x = 'sex', data = data)         #same for sex



sns.distplot(x= data['charges'], color = 'm')         #distplot


sns.distplot(x = np.log(data['charges']),color = 'g')       #same but with log scale


corr = data.corr('pearson')             #correlation and heatmap
mask = np.triu(corr)
sns.heatmap(corr, linewidth = 0.5 , mask = mask , annot = True)
