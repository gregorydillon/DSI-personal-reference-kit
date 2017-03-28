import statsmodels.formula.api as smf
import numpy as np

from load_data import load_clean_from_zip

# Best so far -- .499 error
# Credit to Vy Nguyen
favorite_so_far = '''
SalePrice ~ YearMade +
        Modernity +
        MachineHoursCurrentMeter +
        ProductGroup +
        fiProductClassDesc +
        Enclosure +
        Hydraulics +
        _units*_measurement
'''

def main():
    training_data = load_clean_from_zip('data/tractor_data.zip')


def root_mean_log_squared_error(test_values, predictions):
    '''
        compute the log-root-mean-squared error metric. This function
        doesn't punish being WAY off on some values as highly.
    '''

    log_diff = np.log(predictions + 1) - np.log(test_values + 1)
    return np.sqrt(np.mean(log_diff ** 2))


def model_and_score_from_formula(formula, training_data, test_values):
    '''
        Accept a formula in the StatsModels.formula.api style, some training data and
        some test values that must match the value being predicted by the formula.
    '''
    model = smf.ols(formula=formula, data=training_data).fit()
    predictions = model.predict(test_values)
    score = root_mean_log_squared_error(test_values, predictions)

    return (model, score)


def plot_resid(model, training_data):
    '''
        Given a trained StatsModel linear regression model, plot the residual error
        in a scatter plot as well as a qqplot

        model: a trained StatsModel linear regression model.
        training_data: the input data which was used to train the model.

        returns: the figure upon which the residuals were drawn
    '''
    fig, ax_list = plt.subplots(1, 2)

    y_hat = model.predict(training_data)
    resid = model.outlier_test()['student_resid']

    ax_list[0].scatter(y_hat, resid, alpha=.2)
    ax_list[0].axhline(0, linestyle='--')
    sm.qqplot(resid, line='s', ax=ax_list[1])

    fig.tight_layout()
    return fig
