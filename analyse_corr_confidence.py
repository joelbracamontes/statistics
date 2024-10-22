import numpy as np 
from scipy.stats import pearsonr, norm, probplot, jarque_bera, shapiro, kstest
import statsmodels.api as sm


def Z(r):
    """Fisher Z-Transformation
    Parameters
    ----------
    r:      int or array
            correlation coefficient
    returns:array
            Fisher Z-transformed value of r.
    """
    r = np.asarray(r)  # Ensure r is a numpy array for element-wise operations
    return 0.5*(np.log(r+1) - np.log(1-r))

def inverse_Z(Z):
    """Inverse Fisher Z-transformation
    Z:      int or array
            Fisher Z-transformed value
    returns:array
            Inverse Fisher Z-transformed value of Z.
    """
    Z = np.asarray(Z)  # Ensure Z is a numpy array for element-wise operations
    return (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)

def sigma(N):
    """Standart deviation of the Fisher Z-transformed correlation coefficient
    N:      int
            number of samples
    returns:float
            standart deviation of the Fisher Z-transformed correlation coefficient
    """
    return 1/(N-3)**(0.5)

def t_value(r):
    return r*np.sqrt((len(r)-2)/(1-r**2))

# from Cristina (= 1.96 for 0.05), but should be the same as Z(alpha/2) and it is not
Z_crit = norm.ppf(1-0.05/2) # percent point function, inverse of the cumulative distribution function
#### why is it different from Z(1-(0.05/2)) ??? 

def N_eff_Tint(series):
    """effective number of independent samples from integral timescale, 
    see https://andrewcharlesjones.github.io/journal/21-effective-sample-size.html#:~:text=The%20effective%20sample%20size%20is%20a%20metric%20that,to%20the%20correlation%20and%20redundancy%20between%20the%20samples."""
    n_lag = len(series)-1
    autocorr = sm.tsa.acf(series, nlags=n_lag)
    T_int = 1 + 2*np.sum(autocorr)
    N_eff = len(series)/T_int
    return N_eff

def confid_interval(r, N_eff, Zcrit=1.96):
    """Confidence interval of the correlation coefficient
    r:      int or array
            correlation coefficient
    alpha:  float
            significance level
    N_eff:  int
            effective number of independent samples
    returns:array
            lower bound, upper bound
    """
    return [Z(r) - Z_crit*sigma(N_eff), Z(r) + Z_crit*sigma(N_eff)]

def test_normal_dist(series1):
    """Test if the distribution of the values is normally distributed
    series1:    array
                first array of values
    series2:    array
                second array of values
    returns:    tuple
                test results of the normality test
    """
    # Testing if distribution of values is normally distributed (needed for pearson)
    print('p-value below 0.05 would indicate non-normality')
    print(f'jarqueb: {jarque_bera(series1)}')         # see https://www.statology.org/jarque-bera-test-python/, p-value below 0.05 would indicate skewness
    print(f'shapiro: {shapiro(series1)}')             # see https://www.statology.org/shapiro-wilk-test-python/, p-value below 0.05 would indicate non-normality
    print(f'kstest: {kstest(series1, 'norm')}')       # see https://www.statology.org/kolmogorov-smirnov-test-python/, p-value below 0.05 would indicate non-normality
    return jarque_bera(series1), shapiro(series1),kstest(series1, 'norm')

def plot_corr(corr_dict, plot_dict):
    """
    Plot the correlation between lags and correlations with dynamic plot configuration.
    
    Parameters:
        corr_dict (dict): Dictionary containing 'lags' and 'correlations' keys and 'labels'
                          'lags' represents the lags, 'correlations' represents the correlation values and 'labels' represents the labels for each set of lags and correlations.
        plot_dict (dict): Dictionary to specify plot parameters such as:
                          'title','ylim', 'xlim', 'xlabel', 'ylabel', 'max_lag', 'colors', 'yticks',etc.
    
    Returns:
        fig, ax: The matplotlib figure and axes objects for further customization.

    Example: 
    corr_dict = {'lags': [range(-24*3, 24*3+1)]	, 'correlations': [np.random.uniform(-1,1,145)]	}
    plot_dict = {
        'max_lag': 24*3,
        'xlabel': 'Time shift [days], ocean lags atmosphere',
        'ylabel': r'Correlation with w$_{rms}$',
        'ylim': (-1,1),
        'color': 'r',
        'title': 'Correlation vs Time Lag'
    }
    # Generate the plot
    fig, ax = plot_corr(corr_dict, plot_dict)
    plt.show()
    """
    import matplotlib.pyplot as plt
    
    # Set up figure and axis
    fig, ax = plt.subplots(1,1, figsize=(11, 6))

    # Extract lags and correlation values
    lag_sets            = corr_dict.get('lags', [])
    correlation_sets    = corr_dict.get('correlations', [])
    print(lag_sets, correlation_sets)
    
    # Extract labels and colors (use default if not provided)
    labels = corr_dict.get('labels', [f'Series {i+1}' for i in range(len(lag_sets))])
    colors = plot_dict.get('colors', ['r'] * len(lag_sets))  # Default to red if no colors provided
    
    # Plot each set of lags and correlations
    for i, (lags, correlations) in enumerate(zip(lag_sets, correlation_sets)):
        ax.plot(lags, correlations, color=colors[i], label=labels[i])
    ax.legend()

    # Draw horizontal and vertical reference lines at zero
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='k', linestyle='--')

    # Set axis labels (can be customized in plot_dict)
    xlabel = plot_dict.get('xlabel', 'Time shift [days], ocean lags atmosphere')
    ylabel = plot_dict.get('ylabel', r'Correlation with w$_{rms}$')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom x-ticks at every 24 hours and appropriate labels
    max_lag = plot_dict.get('max_lag', 24*3)        # Default to 3 days if not provided
    x_ticks = list(range(-max_lag, max_lag+1, 24))  # Generate ticks at every 24 hours
    x_labels = [f'{i//24}' if i == 24 else f'{i//24}' for i in x_ticks]
    ax.set_xticks(x_ticks, x_labels)

    # Set custom y-ticks and limits
    ax.set_yticks(plot_dict.get('yticks', [0,0.5]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_ylim(plot_dict.get('ylim', [-0.2,0.8]))
    # Set custom x-limits
    ax.set_xlim(plot_dict.get('xlim',[-24,max_lag]))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5*24)) # Minor ticks at every 12 hours
    # Set optional title
    title = plot_dict.get('title', '')
    ax.set_title(title)
    return fig, ax


def generate_signal_w_noise(length=24*10, noise_scale1=0.2, noise_scale2=0.2, magnitude2=1.5):
    """Generate two time series with a sinusoidal signal and different random noise for each.
    Parameters:
    
    length: int, the length of the time series in hours
    noise_scale1: float, the scale of the random noise to add to the signal1
    noise_scale2: float, the scale of the random noise to add to the signal2
    magnitude2: float, the magnitude of the second signal relative to the first
    
    Returns: pd.Series, pd.Series, the two time series with the second shifted by 12 hours

    Example:
    series1, series2    = generate_signal_w_noise()                     # Generate a sinus signal with random noise
    lags_exp, corrs_exp = ag.lag_correlation(series2, series1, max_lag) # Calculate the lag correlation
    """
    import pandas as pd
    # Set the random seed for reproducibility
    np.random.seed(42)
    # Generate a time index for 10 days at 1-hour intervals
    time_index = pd.date_range(start="2024-10-01", periods=length, freq="H")                                             # maximum time lag in hours
    # Create an underlying signal, e.g., a sine wave with some periodic peaks
    underlying_signal = np.sin(np.linspace(0, 10*np.pi, len(time_index)))  # Sine wave signal

    # Add random noise to the signal for the first series
    noise1  = np.random.normal(loc=0, scale=noise_scale1, size=len(time_index)) # increase scale for more noise
    series1 = pd.Series(underlying_signal + noise1, index=time_index)

    # Create the second series by shifting the underlying signal and adding different random noise
    shift_hours = 12  # Shift by 24 hours (1 day)
    noise2  = np.random.normal(loc=0, scale=noise_scale2, size=len(time_index))
    series2 = magnitude2*pd.Series(np.roll(underlying_signal, shift_hours) + noise2, index=time_index)

    return series1, series2
