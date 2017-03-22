import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('data/balance.csv')
y = df.Balance

pd.tools.plotting.scatter_matrix(df,diagonal='kde')
x = df[df.columns[1:-1]]

#1
# Limit and rating are stongly correlated between each other. thus we should iunclude Rating only
# There is alos sopme correlation between Income and Balance

#2

x.Married = x.Married.map(dict(Yes=1, No=0))
x.Student = x.Student.map(dict(Yes=1, No=0))
x.Gender = x.Gender.map({'Female':1,' Male':0})

#3
dummies = pd.get_dummies(x.Ethnicity.rename(columns = lambda x: "Ethnicity_{}".format(x)))
x = pd.concat([x,dummies],axis=1)
del x['African American']
del x['Ethnicity']

#4

def build_model(y,data, constant=True):
    if constant:
        data = sm.add_constant(data)

    model = sm.OLS(y,data).fit()
    return (model, data)

def build_f_model(data, formula):
    model = smf.ols(data=data, forumla=formula)
    model = model.fit()
    return model

def plot_resid(model,data):
    fig, ax_list = plt.subplots(1, 2)
    y_hat = model.predict(data)
    resid = model.outlier_test()['student_resid']
    ax_list[0].scatter(y_hat,resid)
    ax_list[0].axhline(0, linestyle='--')
    sm.qqplot(resid, line='s', ax=ax_list[1])


model, data = build_model(y,x,True)
model.summary()
plot_resid(model,data)

#5
columns = x.columns.values
#
# model, data = build_model(y,x.filter(['Income', 'Rating', 'Cards', 'Age', 'Education', 'Gender',
#        'Student', 'Married', 'Asian', 'Caucasian']),True)
# model.summary()
# plot_resid(model,data)

# reduced model
model, data = build_model(y,x.filter(['Income', 'Rating', 'Age', 'Student']),True)
print model.summary()
plot_resid(model,data)

plt.hist(y,bins=100)

# This was just us playing to figure out what a good limit was
# 8
test = x.copy()
test['Balance'] = y
test = test[ test['Rating'] >= 230]
tmp = test.copy()
y = test['Balance']
del test['Balance']

#9
model, data = build_model(y,test.filter(['Income', 'Rating', 'Age', 'Student']),True)
print model.summary()
plot_resid(model,data)
#
# for c_name in tmp.columns.values:
#     try:
#         tmp.plot(kind='scatter', y='Balance', x=c_name, edgecolor='none', figsize=(12, 5))
#     except:
#         pass

# EC

income_model = smf.ols(data=tmp, formula='Balance ~ Income').fit()
student_model = smf.ols(data=tmp, formula='Balance ~ Student').fit()
is_model = smf.ols(data=tmp, formula='Balance ~ Income*Student').fit()
print income_model.summary()
print '\n\n'
print student_model.summary()
print '\n\n'
print is_model.summary()

super_model = smf.ols(data=tmp, formula='Balance ~ Income*Student + Rating + Age')
answer = super_model.fit()

original_model = smf.ols(data=tmp, formula='Balance ~ Income + Student + Rating*Age')
answer = original_model.fit()
