import abc
import inspect
import itertools
import re
import typing

REG = "Regression"
BINARY_CLF = "Binary classification"
MULTI_CLF = "Multi-class classification"
MO_BINARY_CLF = "Multi-output binary classification"
MO_REG = "Multi-output regression"
DENS_EST = "Density estimation"


class Dataset(abc.ABC):
    """Base class for all datasets.
    All datasets inherit from this class, be they stored in a file or generated on the fly.
    """

    def __init__(
        self,
        task,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
    ):
        self.task = task
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.sparse = sparse

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def take(self, k: int):
        """Iterate over the k samples."""
        return itertools.islice(self, k)

    @property
    def desc(self):
        """Return the description from the docstring."""
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overridden in order to modify the output of the __repr__ method.
        """

        content = {}
        content["Name"] = self.__class__.__name__
        content["Task"] = self.task
        if isinstance(self, SyntheticDataset) and self.n_samples is None:
            content["Samples"] = "∞"
        elif self.n_samples:
            content["Samples"] = f"{self.n_samples:,}"
        if self.n_features:
            content["Features"] = f"{self.n_features:,}"
        if self.n_outputs:
            content["Outputs"] = f"{self.n_outputs:,}"
        if self.n_classes:
            content["Classes"] = f"{self.n_classes:,}"
        content["Sparse"] = str(self.sparse)

        return content

    def __repr__(self):

        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        out = f"{self.desc}\n\n" + "\n".join(
            k.rjust(l_len) + "  " + v.ljust(r_len)
            for k, v in self._repr_content.items()
        )

        if "Parameters\n    ----------" in self.__doc__:
            params = re.split(
                r"\w+\n\s{4}\-{3,}",
                re.split("Parameters\n    ----------", self.__doc__)[1],
            )[0].rstrip()
            out += f"\n\nParameters\n----------{params}"

        return out


class SyntheticDataset(Dataset):
    """A synthetic dataset."""

    def __repr__(self):
        l_len_prop = max(map(len, self._repr_content.keys()))
        r_len_prop = max(map(len, self._repr_content.values()))
        params = self._get_params()
        l_len_config = max(map(len, params.keys()))
        r_len_config = max(map(len, map(str, params.values())))

        out = (
            "Synthetic data generator\n\n"
            + "\n".join(
                k.rjust(l_len_prop) + "  " + v.ljust(r_len_prop)
                for k, v in self._repr_content.items()
            )
            + "\n\nConfiguration\n-------------\n"
            + "\n".join(
                k.rjust(l_len_config) + "  " + str(v).ljust(r_len_config)
                for k, v in params.items()
            )
        )

        return out

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""
        return {
            name: getattr(self, name)
            for name, param in inspect.signature(self.__init__).parameters.items()  # type: ignore
            if param.kind != param.VAR_KEYWORD
        }

    def _get_params(self):
        """Return the parameters that were used during initialization."""
        return {
            name: getattr(self, name)
            for name, param in inspect.signature(self.__init__).parameters.items()  # type: ignore
            if param.kind != param.VAR_KEYWORD
        }
