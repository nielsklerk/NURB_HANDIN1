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
    data = np.loadtxt(filename)
    features = data[:, :-1]
    features = (features - np.mean(features, axis=0))/np.std(features, axis=0)
    labels = data[:, -1]
    return features, labels


# Make your own implementation of logistic regression
def hypothesis(features, theta):
    z =  theta[:-1] @ features.T + theta[-1]
    return 1 / (1 + np.exp(-z))

def logistic_regression(
    features, labels, feature_combinations, learning_rate=0.1, n_iterations=30):
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
        h = hypothesis(features, theta)
        return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(y)

    def loss_function_grad(features, theta, labels, l=0):
        h = hypothesis(features, theta)
        features_and_constant = np.vstack((features.T, np.ones(features.shape[0])))
        return features_and_constant @ (h-labels) / len(labels) + 2 * l * np.sum(theta)

    theta_values =[]
    loss = np.zeros((n_iterations, len(feature_combinations)))
    for j, combination in enumerate(feature_combinations):
        feature_selection = features[:, [*combination]]
        theta = np.ones(feature_selection.shape[1] + 1)
        for i in range(n_iterations):
            delta_theta = learning_rate*loss_function_grad(feature_selection, theta, labels)
            theta -= delta_theta
            loss[i, j] = loss_function(feature_selection, theta, labels)
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
    predictions = np.round(hypothesis(features[:, feature_columns], theta))
    true_positive = np.sum((predictions == 1) & (labels==1))
    false_positive = np.sum((predictions == 1) & (labels==0))
    true_negative = np.sum((predictions == 0) & (labels==0))
    false_negative = np.sum((predictions == 0) & (labels==1))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_score = 2 * precision * recall / (precision + recall)

    # save txt
    with open(os.path.join(output_dir, "logistic_regression_metrics.txt"), "w") as f:
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
    cost_function, theta_values = logistic_regression(features, labels, feature_combinations, n_iterations=100000, learning_rate=1e-3)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    for i, combination in enumerate(feature_combinations):
        label = f'Feature {combination[0]+1}'
        for other in combination[1:]:
            label += f'+{other+1}'
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
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    names = [r"$\kappa_{CO}$", "Color", "Extended", "Emission line flux"]
    plot_idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    for i, comb in enumerate(itertools.combinations(np.arange(0, 4), 2)):
        ax[plot_idx[i][0], plot_idx[i][1]].scatter(
            features[:, comb[0]], features[:, comb[1]], c=labels
        )
        xlim, ylim = ax[plot_idx[i][0], plot_idx[i][1]].get_xlim(), ax[plot_idx[i][0], plot_idx[i][1]].get_ylim()
        theta = theta_values[i]
        print(theta)
        ax[plot_idx[i][0], plot_idx[i][1]].plot(xlim, -(theta[0] * np.array(xlim) + theta[-1]) / theta[1], "k--")
        ax[plot_idx[i][0], plot_idx[i][1]].set(
            xlabel=names[comb[0]], ylabel=names[comb[1]], xlim=xlim, ylim=ylim
        )
    plt.savefig(os.path.join(output_dir, "fig3c.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
