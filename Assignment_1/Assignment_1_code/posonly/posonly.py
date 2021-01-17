import numpy as np
import util
import sys

### NOTE : You need to complete logreg implementation first!
class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d, dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))

            if np.max(np.abs(prev_theta - self.theta)) < self.eps:
                break

        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self._sigmoid(x.dot(self.theta))

        return y_hat

    def _gradient(self, x, y):
        """Get gradient of J.

        Returns:
            grad: The gradient of J with respect to theta. Same shape as theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        grad = 1 / n * x.T.dot(probs - y)

        return grad

    def _hessian(self, x):
        """Get the Hessian of J given theta and x.

        Returns:
            hess: The Hessian of J. Shape (dim, dim), where dim is dimension of theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        hess = 1 / n * x.T.dot(diag).dot(x)

        return hess

    def _loss(self, x, y):
        """Get the empirical loss for logistic regression."""
        eps = 1e-10
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + eps) + (1 - y) * np.log(1 - hx + eps))

        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    classification_part_a = LogisticRegression(verbose=False, max_iter=1000)
    test_x, test_t = util.load_dataset(test_path, label_col='t', add_intercept=True)
    train_x, train_t = util.load_dataset(train_path, label_col='t', add_intercept=True)
    classification_part_a.fit(train_x, train_t)
    np.savetxt(output_path_true, classification_part_a.predict(test_x))
    util.plot(test_x, test_t, classification_part_a.theta, "question_2_part_a.png")

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    classification_part_b = LogisticRegression(verbose=False, max_iter=1000)
    train_x, train_y = util.load_dataset(train_path, label_col='y', add_intercept=True)
    test_x, test_y = util.load_dataset(test_path, label_col='y', add_intercept=True)
    classification_part_b.fit(train_x, train_y)
    np.savetxt(output_path_naive, classification_part_b.predict(test_x))
    util.plot(test_x, test_t, classification_part_b.theta, "question_2_part_b")

    # Part (f): Apply correction factor using validation set and test on true labels
    classification_part_f = LogisticRegression(verbose=False, max_iter=1000)
    valid_x, valid_y = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    classification_part_f.fit(train_x, train_y)
    #validation and estimating alpha
    x_1 = valid_x[valid_y == 1, -2]
    x_1 = np.reshape(x_1, (len(x_1), 1))
    x_2 = valid_x[valid_y == 1, -1]
    x_2 = np.reshape(x_2, (len(x_2), 1))
    Vpos = np.ones((len(x_2), 1))
    Vpos = np.append(Vpos, x_1, axis=1)
    Vpos = np.append(Vpos, x_2, axis=1)
    alpha = np.mean(classification_part_f.predict(Vpos))
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    np.savetxt(output_path_adjusted, classification_part_f.predict(test_x))
    util.plot(test_x, test_t, classification_part_f.theta, 'question_2_part_f.png', correction=alpha)
    # *** END CODER HERE



if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
