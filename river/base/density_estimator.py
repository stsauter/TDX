import abc

import pandas as pd

from . import estimator


class DensityEstimator(estimator.Estimator):
    """A density estimator."""

    @property
    def _supervised(self):
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> "DensityEstimator":
        """Update the model.
        Parameters
        ----------
        x
            A dictionary of features.
        Returns
        -------
        self
        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predicts the density of a set of features `x`.
        Parameters
        ----------
        x
            A dictionary of features.
        Returns
        -------
        The prediction.
        """


class MiniBatchDensityEstimator(DensityEstimator):
    """A density estimator that can operate on mini-batches."""

    @abc.abstractmethod
    def learn_many(self, X: pd.DataFrame, **kwargs) -> "MiniBatchDensityEstimator":
        """Update the model with a mini-batch of features `X`.
        Parameters
        ----------
        X
            A dataframe of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.
        Returns
        -------
        self
        """

    @abc.abstractmethod
    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the outcome for each given sample.
        Parameters
        ---------
        X
            A dataframe of features.
        Returns
        -------
        The predicted outcomes.
        """
