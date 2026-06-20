import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

# Question 3: Spiral and elliptical galaxies


def load_and_prepare_data(filename):
    """
    Load the galaxy dataset and prepare the feature matrix and binary class labels

    Returns
    -------
    features : ndarray, shape (m, 4)
        Matrix containing the four rescaled input features
        Each feature should have mean 0 and standard deviation 1

    labels : ndarray, shape (m,)
        A value of 1 corresponds to spiral galaxies
        A value of 0 corresponds to elliptical galaxies
    """
    # Loading the data
    data = np.loadtxt(filename)

    # Splitting the data in features and labels
    features = data[:, :-1]
    labels = data[:, -1]

    # Apply standardization to the features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features, labels


# Make your own implementation of logistic regression
def hypothesis(features, theta):
    # Apply weights and bias
    z = theta[:-1] @ features.T + theta[-1]

    # Return the result using sigmoid function
    return 1 / (1 + np.exp(-z))

def downhillsimplex(function, points, n_iters=100):
    def sort_array(
                    arr: np.ndarray,
                    inplace: bool = False,
                    index = False
                    ) -> np.ndarray:
        """
        Sort a 1D array using merge sort

        Parameters
        ----------
        arr : ndarray
            Input array to be sorted
        inplace : bool, optional
            If True, sort the array in-place
            If False, return a sorted copy

        Returns
        -------
        sorted_arr : ndarray
            Sorted array (same shape as arr)

        """
        def roll(array, shift):
            shifted_array = np.empty_like(array)
            shifted_array[:shift] = array[-shift:]
            shifted_array[shift:] = array[:-shift]
            return shifted_array
        
        # Make an copy if not inplace
        if inplace:
            sorted_arr = arr
        else:
            sorted_arr = arr.copy()

        N = len(sorted_arr)
        step = 1
        index_array = np.arange(N)

        while step < N:
            # Loop over pairs of subarrays
            for i in range(0, N, step*2):
                # Set pointers for 2 adjacent subarrays
                l = i
                r = min(i + step, N)
                end = min(i + 2*step, N)

                # Loop over the elements of both subarrays
                while l < r and r < end:
                    # If the left <=  the right, the left is in the correct place
                    if sorted_arr[l] <= sorted_arr[r]:
                        # increase left pointer
                        l += 1

                    # If the left > the right, the array is rolled to the correct order
                    else:
                        sorted_arr[l:r+1] = roll(sorted_arr[l:r+1], 1)
                        index_array[l:r+1] = roll(index_array[l:r+1], 1)
                        # Increase both pointers
                        l += 1
                        r += 1
            # Double step size
            step *= 2

        if index:
            return index_array
        return sorted_arr

    function_values = np.zeros(n_iters)
    for i in range(n_iters):
        new_point_added = False

        # Sort the vertices
        points = points[sort_array([function(point) for point in points], index=True)]

        # Find the centroid
        centroid = np.mean(points[:-1], axis=0)

        f_0 = function(points[0])
        f_N = function(points[-1])
        function_values[i] = f_0
        
        # Reflect worst vertext
        point_try = 2 * centroid - points[-1]
        f_try = function(point_try)
        if f_0 <= f_try and f_try < f_N:
            points[-1] = point_try
            new_point_added = True
        elif f_try < f_0:
            # Extend the reflected point
            point_exp = 2 * point_try - centroid
            if function(point_exp) < f_try:
                points[-1] = point_exp
                new_point_added = True
            else:
                points[-1] = point_try
                new_point_added = True

        else:
            # Contract the worst vertex to the centroid
            point_try = 0.5 * (centroid + points[-1])
            if function(point_try) < f_N:
                points[-1] = point_try
                new_point_added = True
        
        if not new_point_added:
            # Shrink all vertices to the best vertex
            points[1:] = 0.5 * (points[0] + points[1:])
    
    # Sort the points of the last vertex and return the best point
    points = points[sort_array([function(point) for point in points], index=True)]
    return points[0], function_values


def logistic_regression(
    features, labels, feature_combinations, n_iterations=30
):
    """
    This function should select a chosen set
    of input feature columns, then fit a logistic regression model to classify
    galaxies as spirals or ellipticals.

    Parameters
    ----------
    features : ndarray, shape (m, 4)
        Rescaled feature matrix

    labels : ndarray, shape (m,)
        1 corresponds to spiral galaxies
        0 corresponds to elliptical galaxies

    feature_combinations : list of tuple
        example: [(0, 1), (0, 2)]

    learning_rate : float, optional
        Step size used in gradient descent

    n_iterations : int, optional
        Number of minimisation iterations

    Returns
    -------
    cost_function : ndarray, shape (n_iterations, n_combinations)
        Cost function values for every iteration and every feature combination

    theta_values : list of ndarray
        Best-fit parameters for each feature combination"""

    def loss_function(features, theta, y):
        # Find the predicted percentages using the model
        h = hypothesis(features, theta)

        # Return logistic loss function
        return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(y)

    theta_values = []
    loss = np.zeros((n_iterations, len(feature_combinations)))
    for j, combination in enumerate(feature_combinations):
        feature_selection = features[:, [*combination]]

        # Creating the verteces for the starting simplex
        point = np.ones(feature_selection.shape[1] + 1)
        starting_points = [point]
        for i in range(feature_selection.shape[1] + 1):
            point[i] += 1
            starting_points.append(point.copy())
            point[i] -= 1
        starting_points = np.array(starting_points)

        # Using downhill simplex for minimization for the loss function
        function_to_minimize = lambda x: loss_function(feature_selection, x, labels)
        theta, loss[:, j] = downhillsimplex(function_to_minimize, starting_points, n_iters=n_iterations)

        # Store the the best theta values for this combination
        theta_values.append(theta)

    return loss, theta_values


def test_logistic_regression(features, labels, theta, feature_columns, output_dir):
    """
    Compute the number of true/false positives/negatives, as well as the F1 score, and save them for your report

    Parameters
    ----------
    features : ndarray, shape (m, 4)
        Rescaled feature matrix

    labels : ndarray, shape (m,)
        True binary class labels

    theta : ndarray
        Logistic regression parameters returned by logistic_regression()

    feature_columns : list or tuple
        Feature columns corresponding to parameters used by the trained model
    output_dir : str
        Directory where to save the results

    Returns
    -------
    predictions : ndarray, shape (m,)
        Predicted class labels

    true_positive : int

    false_positive : int

    true_negative : int

    false_negative : int

    f1_score : float
    """
    # Find the predicted values
    predictions = np.round(hypothesis(features[:, feature_columns], theta))

    # Find the number in the econfusion matrix
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    true_negative = np.sum((predictions == 0) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))

    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    # Use precision and recall to calculate the F1-score
    f1_score = 2 * precision * recall / (precision + recall)

    # save txt
    with open(os.path.join(output_dir, "logistic_regression_metrics.txt"), "w") as f:
        f.write(f"True Positives: {true_positive}\n")
        f.write(f"True Positives: {true_positive}\n")
        f.write(f"False Positives: {false_positive}\n")
        f.write(f"True Negatives: {true_negative}\n")
        f.write(f"False Negatives: {false_negative}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")

    return (
        predictions,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        f1_score,
    )


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Problem 3.a
    features, labels = load_and_prepare_data("Data/galaxy_data.txt")
    np.savetxt(
        os.path.join(output_dir, "galaxy_data_scaled.txt"),
        features,
        header="kappa_CO Color Extended Emission_line_flux",
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].hist(features[:, 0], bins=20)
    ax[0, 0].set(ylabel="N", xlabel=r"$\kappa_{CO}$")
    ax[0, 1].hist(features[:, 1], bins=20)
    ax[0, 1].set(xlabel="Color")
    ax[1, 0].hist(features[:, 2], bins=20)
    ax[1, 0].set(ylabel="N", xlabel="Extended")
    ax[1, 1].hist(features[:, 3], bins=20)
    ax[1, 1].set(xlabel="Emission line flux")
    plt.savefig(os.path.join(output_dir, "fig3a.png"), dpi=300)
    plt.close()

    # Problem 3.b
    feature_combinations = [*itertools.combinations(np.arange(0, 4), 2)]
    cost_function, theta_values = logistic_regression(
        features, labels, feature_combinations, n_iterations=100)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    for i, combination in enumerate(feature_combinations):
        label = f"Feature {combination[0]+1}"
        for other in combination[1:]:
            label += f"+{other+1}"
        ax.plot(np.arange(0, len(cost_function)), cost_function[:, i], label=label)
    # ...........
    ax.set(xlabel="Number of iterations", ylabel="Cost function")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, "fig3b.png"), dpi=300)
    plt.close()

    # Problem 3.c

    # For every pair of features, plot the two features against each other and indicate the decision boundary
    (
        predictions,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        f1_score,
    ) = test_logistic_regression(
        features,
        labels,
        theta=theta_values[0],
        feature_columns=feature_combinations[0],
        output_dir=output_dir,
    )  # REPLACE with the parameters corresponding to the trained model using features 1 and 2; then repeat for the other feature combinations
    for i in range(len(feature_combinations)):
        (
        predictions,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        f1_score,
        ) = test_logistic_regression(
            features,
            labels,
            theta=theta_values[i],
            feature_columns=feature_combinations[i],
            output_dir=output_dir,
        )
        with open(os.path.join(output_dir, f"logistic_regression_metrics{i}.txt"), "w") as f:
            f.write(f"{feature_combinations[i][0]}+{feature_combinations[i][1]} & {true_positive} & {false_positive} & {true_negative} & {false_negative} & {f1_score:.4f}")

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    names = [r"$\kappa_{CO}$", "Color", "Extended", "Emission line flux"]
    plot_idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    for i, comb in enumerate(itertools.combinations(np.arange(0, 4), 2)):
        # Plot the features
        ax[plot_idx[i][0], plot_idx[i][1]].scatter(
            features[:, comb[0]], features[:, comb[1]], c=labels
        )
        
        # Store the x and y limits to use after plotting the decission boundary to use later
        xlim, ylim = (
            ax[plot_idx[i][0], plot_idx[i][1]].get_xlim(),
            ax[plot_idx[i][0], plot_idx[i][1]].get_ylim(),
        )
        
        # Extract theta for this combination
        theta = theta_values[i]

        # Plot the decision boundary
        ax[plot_idx[i][0], plot_idx[i][1]].plot(
            xlim, -(theta[0] * np.array(xlim) + theta[-1]) / theta[1], "k--"
        )

        # Set the labels and the limits
        ax[plot_idx[i][0], plot_idx[i][1]].set(
            xlabel=names[comb[0]], ylabel=names[comb[1]], xlim=xlim, ylim=ylim
        )
    plt.savefig(os.path.join(output_dir, "fig3c.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
