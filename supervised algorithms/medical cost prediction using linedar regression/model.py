from sklearn.preprocessing import LabelEncoder              # we want to encode feature values
label=LabelEncoder()                                        #sex, smoker, region these three column are of object type
label.fit(data.sex.drop_duplicates())
data.sex=label.transform(data.sex)
label.fit(data.smoker.drop_duplicates())
data.smoker=label.transform(data.smoker)
label.fit(data.region.drop_duplicates())
data.region=label.transform(data.region)



x=data.drop(['charges'],axis=1)                                         # preparing training set 
y=data['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)                #spliting the datasets into two different sets for training and testing



#we're considering five regression models that are already imported

linreg = LinearRegression()
rfc = RandomForestRegressor(n_estimators = 500 , max_features = 6 , max_depth = 10 , min_samples_leaf = 6)
abr = AdaBoostRegressor(learning_rate = 0.1 )
gbr = GradientBoostingRegressor()
xgb = XGBRegressor(gamma = 0.5)




models = [linreg, rfc, abr, gbr, xgb]

#creating a function to find the optimal model 
def training(model):
    score_dict =[]
    error_dict =[]
    for m in model:
        m.fit(x_train,y_train)
        score = m.score(x_test, y_test)
        y_pred = m.predict(x_test)
        model_rmse = mean_squared_error(y_test, y_pred, squared = False)
        print("for the model",m, "the acceuracy & error are ", score, model_rmse)
        score_dict.append(score)
        error_dict.append(model_rmse)
    print("the final Results are :\n")
    print(score_dict)
    print(error_dict)
    
    
   
# Calling the function we just built 
training(models)

#traing the best fit model

RFC = RandomForestRegressor(n_estimators = 500 , max_features = 6 , max_depth = 10 , min_samples_leaf = 6)
RFC.fit(x_train , y_train)



score = RFC.score(x_test, y_test)


print(score)
