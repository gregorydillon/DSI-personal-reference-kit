import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

OUTPUT_DIR = 'output/'

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

    generate_initial_report(balance_data, 3, "balance_report")

    # Prepare for regression, balance_series == y, training_data == x
    training_data = balance_data[balance_data.columns[1:-1]]  # Slice off id and balance
    balance_series = balance_data.Balance

    # reduced model
    relevant_cols = ['Income', 'Rating', 'Age', 'Student']
    build_regression_report('inc_age_rating_student', relevant_cols, training_data, balance_series)

    relevant_cols = ['Rating']
    build_regression_report('rating_only', relevant_cols, training_data, balance_series)

    # Trimming due to hella 0's, we found a reasonable cuttoff in rating at 230
    # (by looking at the scatter plot of it.)
    trimmed_training = training_data.copy()
    trimmed_training['Balance'] = balance_series
    trimmed_training = trimmed_training[trimmed_training['Rating'] >= 230]
    trimmed_balance_series = trimmed_training['Balance']
    del trimmed_training['Balance']

    relevant_cols = ['Income', 'Rating', 'Age', 'Student']
    build_regression_report('trimmed', relevant_cols, trimmed_training, trimmed_balance_series)


def generate_initial_report(df, plot_size_scalar, report_name):
    '''
        Print some initial summary data about our dataframe, plot a scatter_matrix,
        and several box-plots / violin plots.
    '''
    # make the pdf
    report = PdfPages(OUTPUT_DIR + report_name + '.pdf')

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
    '''
        Provided a dataframe, create a figure which has for each column:
         * a violin plot
         * a box plot

         return that figure.
    '''
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


def build_model(y, x, add_constant=True):
    '''
        Build a linear regression model from the provided data

        Provided:
        y: a series or single column dataframe holding our solution vector for linear regression
        x: a dataframe that runs parallel to y, with all the features for our linear regression
        add_constant: a boolean, if true it will add a constant row to our provided x data. Otherwise
                      this method assumes you've done-so already, or do not want one for some good reason

        Return: a linear regression model from StatsModels and the data which was used to train the model.
                If add_constant was true this will be a new dataframe, otherwise it will be x
    '''
    if add_constant:
        x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    return (model, x)


def plot_resid(model, x):
    '''
        Given a trained StatsModel linear regression model, plot the residual error
        in a scatter plot as well as a qqplot

        model: a trained StatsModel linear regression model.
        x: the input data which was used to train the model.

        returns: the figure upon which the residuals were drawn
    '''
    fig, ax_list = plt.subplots(1, 2)

    y_hat = model.predict(x)
    resid = model.outlier_test()['student_resid']

    ax_list[0].scatter(y_hat, resid, alpha=.2)
    ax_list[0].axhline(0, linestyle='--')
    sm.qqplot(resid, line='s', ax=ax_list[1])

    fig.tight_layout()
    return fig


def build_regression_report(report_name, relevant_col_names, training_data, training_answers):
    '''
        Given a report_name, a list of columns to regress on, and the required training_data
        create a regression model using StatsModel. Plot the residuals and a QQ plot and write
        the model.summary() to the report.

        report_name: The name of the pdf
        relevant_col_names: a list with the columns you care about in training_data
        training_data: the training set
        training_answers: y, assumed to be parallel to training_data
    '''
    report = PdfPages(OUTPUT_DIR + report_name + '.pdf')
    reduced_dataset = training_data.filter(relevant_col_names)

    model, data = build_model(training_answers, reduced_dataset)
    summary_text = model.summary()
    with open(OUTPUT_DIR + report_name + ".txt", "w") as text_file:
        text_file.write(str(summary_text))

    resid_fig = plot_resid(model, data)
    report.savefig(resid_fig)
    report.close()


if __name__ == '__main__':
    main()
