import numpy as np
import scipy.stats as st


def generate_visit_samples(n_visits, distribution_ctr):
    '''
        Provided a number of visits and a distribution CTR (the assumed true value
        for Click Through Rate) return a numpy array with n_visits items. Each of
        these items represents a visit to your website and the value will be
        1 for this visitor "clicked through" or 0 for this visitor did not click through.

        Simulated values are created using a Bernoulli distribution using distribution_ctr
        as p.

        n_visits (int): The number of visits to simulate
        distribution_ctr (float): The probability used as p in our Bernoulli distribution

        returns (array like): the samples
    '''
    return st.bernoulli.rvs(distribution_ctr, size=n_visits)
