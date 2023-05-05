import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.2,
            (1, 0): 0.3,
            (1, 1): 0.3
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.1,
            (0, 1): 0.1,
            (1, 0): 0.4,
            (1, 1): 0.4
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.04,
            (0, 0, 1): 0.04,
            (0, 1, 0): 0.16,
            (0, 1, 1): 0.16,
            (1, 0, 0): 0.06,
            (1, 0, 1): 0.06,
            (1, 1, 0): 0.24,
            (1, 1, 1): 0.24,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        
        for x_value, y_value in X_Y.keys():
            if not np.isclose(X.get(x_value) * Y.get(y_value), X_Y.get((x_value, y_value))):
                return True
        
        return False

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################

        for x_value, y_value, c_value in X_Y_C.keys():
            if not np.isclose(X_C.get((x_value, c_value)) * Y_C.get((y_value, c_value)) / C.get(c_value), 
                            X_Y_C.get((x_value, y_value, c_value))):
                return False
        
        return True

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    
    poisson_pmf = (rate ** k) * (np.e ** (- rate)) / (np.math.factorial(k))

    log_p = np.log(poisson_pmf) if rate != 0 else 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    likelihoods = [sum([poisson_log_pmf(sample, rate_i) 
                        for sample in samples]) 
                        for rate_i in rates]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    
    max_likelihood = max(likelihoods)

    rate = rates[likelihoods.index(max_likelihood)]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    
    mean = sum(samples) / len(samples)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    if std != 0:
        p = 1 / np.sqrt(2 * np.pi * np.square(std))
        p *= np.e ** (-(1 / 2) * (np.square(x - mean) / np.square(std)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        # Original class value 
        self.class_value = class_value

        # Original data 
        self.dataset = dataset

        # A part of the dataset where the last column values (label) equals to class_value.
        # I use this variable at the calc_mean() and clac_std().
        self.rows_with_class_value_as_label = dataset[dataset[:, -1] == class_value]

        self.mean = None
        self.calc_mean()

        self.std = None
        self.calc_std()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_mean(self):
        """
        Computed the mean from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.mean = np.mean(self.rows_with_class_value_as_label, 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_std(self):
        """
        Computed the std from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.std = np.std(self.rows_with_class_value_as_label, 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        prior = self.rows_with_class_value_as_label.shape[0] / self.dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        likelihood = 1

        for column_index in range(x.shape[0] - 1):
            likelihood *= normal_pdf(x[column_index], self.mean[column_index], self.std[column_index])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        pred = 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    count_currect_predictions = 0

    for line_index in range(test_set.shape[0]):
        count_currect_predictions += 1 if map_classifier.predict(test_set[line_index, :]) == test_set[line_index, :][-1] else 0

    acc = count_currect_predictions / test_set.shape[0]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    d = cov.shape[1]  # Num of features.

    pdf = (1 / np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))) * \
            (np.e ** ((- 1 / 2) * \
                       np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        # Original class value 
        self.class_value = class_value

        # Original data 
        self.dataset = dataset

        # A part of the dataset where the last column values (label) equals to class_value.
        # I use this variable at the calc_mean() and clac_std().
        self.rows_with_class_value_as_label = dataset[dataset[:, -1] == class_value]

        self.mean = None
        self.calc_mean()

        self.cov = None
        self.calc_cov()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_mean(self):
        """
        Computed the mean from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.mean = np.mean(self.rows_with_class_value_as_label[:, :-1], 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_cov(self):
        """
        Computed the cov from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.cov = np.cov(self.rows_with_class_value_as_label[:, :-1], rowvar=False)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        prior = self.rows_with_class_value_as_label.shape[0] / self.dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        likelihood = multi_normal_pdf(x[:-1], self.mean, self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the prior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        pred = 0 if self.ccd0.get_prior() > self.ccd1.get_prior() else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the likelihood probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        pred = 0 if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x) else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        # Original class value 
        self.class_value = class_value

        # Original data 
        self.dataset = dataset

        # A part of the dataset where the last column values (label) equals to class_value.
        # I use this variable at the calc_mean() and clac_std().
        self.rows_with_class_value_as_label = dataset[dataset[:, -1] == class_value]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        prior = self.rows_with_class_value_as_label.shape[0] / self.dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        likelihood = 1

        # With Laplace smoothing.

        for column_index in range(x.shape[0] - 1):
            n_ij = self.rows_with_class_value_as_label[self.rows_with_class_value_as_label
                            [:, column_index] == x[column_index]].shape[0]
            
            v_j = np.unique(self.dataset[:, column_index]).shape[0]
            
            likelihood *= ((n_ij + 1) / (self.rows_with_class_value_as_label.shape[0] + v_j)) \
                            if x[column_index] in self.dataset[:, column_index] else EPSILLON

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # Implement the function.                                           #
        ###########################################################################
        pred = 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################
        count_currect_predictions = 0

        for line_index in range(test_set.shape[0]):
            count_currect_predictions += 1 if self.predict(test_set[line_index, :]) == test_set[line_index, :][-1] else 0

        acc = count_currect_predictions / test_set.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


