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
