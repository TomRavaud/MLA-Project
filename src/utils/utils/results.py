"""
Some functions to plot the results of the experiments.
(central tendency, spread, significance test)
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


# Import custom modules and packages
import params.results


def plot_central_tendency_and_spread(score_dict: dict, percentile: float=10):
    """Plot the central tendency and the spread of the scores.

    Args:
        score_dict (dict): Dictionary of scores.
        percentile (float, optional): Percentile for the spread.
        Defaults to 10.
    """
    
    # Open a figure
    plt.figure() 

    for method, scores in score_dict.items():
        
        # Compute the central tendency (mean)
        mean_scores = np.mean(scores, axis=0)
        
        # Compute the spread (percentile)
        lower_bound = np.percentile(scores,
                                    q=percentile,
                                    axis=0)
        upper_bound = np.percentile(scores,
                                    q=100-percentile,
                                    axis=0)
        
        # Get the number of epochs
        nb_epochs = len(mean_scores)
        
        
        # Plot the central tendency and the spread
        plt.plot(range(nb_epochs),
                 mean_scores,
                 params.results.MARKER,
                 markersize=params.results.MARKER_SIZE,
                 label=method)
        plt.fill_between(range(nb_epochs),
                         lower_bound,
                         upper_bound,
                         alpha=params.results.OPACITY)

    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    


def significance_test(score_dict: dict, confidence_level: float=0.05):
    """Perform a significance test between the scores of two methods.
    (Welch's t-test)

    Args:
        score_dict (dict): Dictionary of scores (it should contain
        exactly two methods or the first two methods will be used)
        confidence_level (float, optional): Confidence level for the
        significance test. Defaults to 0.05.
    """
    
    # Get the scores for each method
    data1 = score_dict[list(score_dict.keys())[0]]
    data2 = score_dict[list(score_dict.keys())[1]]
    
    # Get the number of epochs
    nb_epochs = data1.shape[1]
    
    # Initialize the array of significative differences
    sign_diff = np.zeros((nb_epochs))
    
    # Compute the p-value for each epoch
    for epoch in range(nb_epochs):
        
        # Perform the significance test
        _, p = scipy.stats.ttest_ind(data1[:, epoch],
                                     data2[:, epoch],
                                     equal_var=False)
        
        # Store the result
        sign_diff[epoch] = p < confidence_level
    
    
    # Get the indices of the significative differences
    id_sign_diff = np.argwhere(sign_diff == 1)
    
    # When working with accuracies, plot the dots a little above 100%
    y = 100
    
    # Plot the dots
    plt.scatter(np.arange(nb_epochs)[id_sign_diff],
                y*1.05*np.ones([id_sign_diff.size]),
                s=params.results.SCATTER_SIZE,
                c="k",
                marker="o",
                label="Significative difference")
    
    plt.legend()
    


if __name__ == "__main__":
    
    # Set some parameters
    nb_epochs = 50
    nb_seeds = 15
    
    # Generate dummy data
    data1 = 80*np.random.rand(nb_seeds, nb_epochs)
    data2 = 60*np.random.rand(nb_seeds, nb_epochs)
    
    # Create the score dictionary
    score_dict = {"Method 1": data1, "Method 2": data2}
    
    # Plot the results
    plot_central_tendency_and_spread(score_dict,
                                     percentile=params.results.PERCENTILE)
    
    significance_test(score_dict,
                      confidence_level=params.results.CONFIDENCE_LEVEL)
    
    plt.show()
