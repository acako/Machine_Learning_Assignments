import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        Xtranspose = np.transpose(X)
        XtransposeX = np.dot(Xtranspose, X)
        XtransposeY = np.dot(Xtranspose, y)
        self.theta = np.linalg.solve(XtransposeX, XtransposeY)
        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        alpha = 0.000002
        iterations = [100, 1000, 10000]

        lenY = len(y)
        costs = []  # np.zeros(iterations)
        thetas = []  # np.zeros((iterations, 2))

        self.fit(X, y)
        tmpTheta = self.theta  # np.random.randn(2, 1)

        for ctr in range(iterations[0]):
            tmp = np.dot(X, tmpTheta)
            # delta = tmp - y
            # XTransposeDelta = (X.T.dot((tmp - y)))
            tmpTheta = tmpTheta - (1 / lenY) * alpha * (X.T.dot((tmp - y)))
            thetas.append(tmpTheta.T)

            tmp = np.dot(X, tmpTheta)
            cost = (1 / 2 * lenY) * np.sum(np.square(tmp - y))
            costs.append(cost)

        self.theta = tmpTheta
        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        alpha = 0.000002
        iterations = [100, 1000, 10000]

        lenY = len(y)
        costs = []  # np.zeros(iterations)
        thetas = []  # np.zeros((iterations, 2))

        self.fit(X, y)
        tmpTheta = self.theta  # np.random.randn(2, 1)

        for ctr in range(iterations[0]):
            tmp_rand = np.random.randint(0, lenY)
            tmp_X = X[tmp_rand, :].reshape(1, X.shape[1])
            tmp_Y = y[tmp_rand].reshape(1, 1)
            tmp = np.dot(tmp_X, tmpTheta)
            # delta = tmp - y
            # XTransposeDelta = (X.T.dot((tmp - y)))
            tmpTheta = tmpTheta - (1 / lenY) * alpha * (tmp_X.T.dot((tmp - tmp_Y)))
            thetas.append(tmpTheta.T)

            tmp = np.dot(tmp_X, tmpTheta)
            cost = (1 / 2 * lenY) * np.sum(np.square(tmp - tmp_Y))
            costs.append(cost)

        self.theta = tmpTheta
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        shapedX = np.reshape(X[:, 1], (-1, 1))
        to_return = []
        for ctr in range(k + 1):
            shapedXExpCtr = shapedX ** ctr
            to_return.append(shapedXExpCtr)

        to_return = np.concatenate(to_return, axis=1)
        return to_return
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        shapedX = np.reshape(X[:, 1], (-1, 1))
        createdPoly = self.create_poly(k, X)
        generatedSin = np.sin(shapedX)
        to_return = np.concatenate([generatedSin, createdPoly], axis=1)
        return to_return
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        to_return = np.dot(X, self.theta)
        return to_return
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        linearModel = LinearModel()
        if sine == True:
            model_trained_sin = linearModel.create_sin(k, train_x)
            model_plot_sin = linearModel.create_sin(k, plot_x)

            if "part_three_GD" in filename:
                linearModel.fit_GD(model_trained_sin, train_y)
            elif "part_three_SGD" in filename:
                linearModel.fit_SGD(model_trained_sin, train_y)
            else:
                linearModel.fit(model_trained_sin, train_y)

            plot_y = linearModel.predict(model_plot_sin)
        else:
            model_trained_poly = linearModel.create_poly(k, train_x)
            model_plot_poly = linearModel.create_poly(k, plot_x)

            if "part_three_GD" in filename:
                linearModel.fit_GD(model_trained_poly, train_y)
            elif "part_three_SGD" in filename:
                linearModel.fit_SGD(model_trained_poly, train_y)
            else:
                linearModel.fit(model_trained_poly, train_y)

            plot_y = linearModel.predict(model_plot_poly)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def run_exp_normal_gd_sgd(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        linearModel = LinearModel()

        model_trained_poly = linearModel.create_poly(k, train_x)
        model_plot_poly = linearModel.create_poly(k, plot_x)

        linearModel.fit_GD(model_trained_poly, train_y)
        plot_gd = linearModel.predict(model_plot_poly)

        linearModel.fit_SGD(model_trained_poly, train_y)
        plot_sgd = linearModel.predict(model_plot_poly)

        linearModel.fit(model_trained_poly, train_y)
        plot_lm = linearModel.predict(model_plot_poly)

        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_gd, plot_sgd, plot_lm, label='k=%d' % k)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, False, [3], 'linear_regression_part_two.png')
    run_exp(train_path, False, [3], 'linear_regression_part_three_GD.png')
    run_exp(train_path, False, [3], 'linear_regression_part_three_SGD.png')
    run_exp(train_path, False, [3, 5, 10, 20], 'linear_regression_part_four.png')
    run_exp(train_path, True, ks=[0, 1, 2, 3, 5, 10, 20], filename='linear_regression_part_five.png')
    run_exp(small_path, False, ks=[1, 2, 5, 10, 20], filename='linear_regression_part_six.png')


    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='train.csv',
         small_path='small.csv',
         eval_path='test.csv')
