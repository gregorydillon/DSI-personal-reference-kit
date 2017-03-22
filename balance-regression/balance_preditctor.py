import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main():
    # Grab the data
    balance_data = pd.read_csv('data/balance.csv')

    # Clean the data
    balance_data.Married = balance_data.Married.map(dict(Yes=1, No=0))
    balance_data.Student = balance_data.Student.map(dict(Yes=1, No=0))
    balance_data.Gender = balance_data.Gender.map({'Female':1, ' Male':0})
    balance_data.Ethnicity = balance_data.Ethnicity.map({
        'Caucasian': 0,
        'Asian': 1,
        'African American': 2
    })

    generate_initial_report(balance_data, 3)

    # Prepare for regression, balance_series == y, training_data == x
    training_data = balance_data[balance_data.columns[1:-1]]  # Slice off id and balance
    balance_series = balance_data.Balance


def generate_initial_report(df, plot_size_scalar):
    '''
        Print some initial summary data about our dataframe, plot a scatter_matrix,
        and several box-plots / violin plots.
    '''
    # Generate text from pd summary data
    report = PdfPages('balance_report.pdf')
    text_fig, ax_list = plt.subplots()
    text_fig.text(.1, .1, df.info())
    text_fig.text(.2, .2, df.head(5))
    report.savefig(text_fig)

    # Generate a scatter matrix
    c_count = len(df.columns)
    scatter_fig, ax_list = plt.subplots(figsize=(c_count*plot_size_scalar, c_count*plot_size_scalar))
    pd.tools.plotting.scatter_matrix(df, diagonal='kde', ax=ax_list)
    report.savefig(scatter_fig)

    # Generate the box/violin overlay plot
    box_fig = box_plots(df, plot_size_scalar, plot_size_scalar*1.5)
    report.savefig(box_fig)
    report.close()


def box_plots(df, plot_width, plot_height):
    c_count = len(df.columns)

    fig, ax_list = plt.subplots(int(len(df.columns) / 2), 2,
        figsize=(2 * plot_width, c_count * plot_height)
    )

    for col_name, ax in zip(df, ax_list.ravel()):
        series = df[col_name]
        ax.violinplot(series, showmeans=False, showmedians=True)
        ax.boxplot(series)
        ax.set_xlabel(col_name)

    plt.tight_layout()
    return fig


def uncategoriezed():
    #3
    dummies = pd.get_dummies(training_data.Ethnicity.rename(columns = lambda training_data: "Ethnicity_{}".format(training_data)))
    training_data = pd.concat([training_data,dummies],axis=1)
    del training_data['African American']
    del training_data['Ethnicity']

    #4

    def build_model(balance_series,data, constant=True):
        if constant:
            data = sm.add_constant(data)

        model = sm.OLS(balance_series,data).fit()
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


    model, data = build_model(balance_series,training_data,True)
    model.summary()
    plot_resid(model,data)

    #5
    columns = training_data.columns.values
    #
    # model, data = build_model(balance_series,training_data.filter(['Income', 'Rating', 'Cards', 'Age', 'Education', 'Gender',
    #        'Student', 'Married', 'Asian', 'Caucasian']),True)
    # model.summary()
    # plot_resid(model,data)

    # reduced model
    model, data = build_model(balance_series,training_data.filter(['Income', 'Rating', 'Age', 'Student']),True)
    print model.summary()
    plot_resid(model,data)

    plt.hist(balance_series,bins=100)

    # This was just us playing to figure out what a good limit was
    # 8
    test = training_data.copy()
    test['Balance'] = balance_series
    test = test[ test['Rating'] >= 230]
    tmp = test.copy()
    balance_series = test['Balance']
    del test['Balance']

    #9
    model, data = build_model(balance_series,test.filter(['Income', 'Rating', 'Age', 'Student']),True)
    print model.summary()
    plot_resid(model,data)
    #
    # for c_name in tmp.columns.values:
    #     try:
    #         tmp.plot(kind='scatter', balance_series='Balance', training_data=c_name, edgecolor='none', figsize=(12, 5))
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


if __name__ == '__main__':
    main()
