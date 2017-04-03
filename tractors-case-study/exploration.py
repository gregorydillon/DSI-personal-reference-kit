import matplotlib.pyplot as plt


def histograms(df, plot_width, plot_height):
    '''
        Provided a dataframe, create a figure which has for each column:
         * A histogram of the unique values per figure

         return that figure.
    '''
    c_count = len(df.columns)

    fig, ax_list = plt.subplots(c_count, 1, figsize=(plot_width, c_count * plot_height))

    for col_name, ax in zip(df, ax_list.ravel()):
        series = df[col_name]

        if series.dtype == float:
            series.hist(ax=ax, title=col_name, normed=True)
        else:
            counts = series.value_counts(normalize=True)
            counts.plot(kind='bar', title=col_name, ax=ax)

        plt.tight_layout()
        ax.set_xlabel(col_name)

    plt.show()

    return fig
