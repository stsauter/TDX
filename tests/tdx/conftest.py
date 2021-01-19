import os
import pytest


class Helpers:
    @staticmethod
    def get_test_file_path(file_name):
        path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(path, file_name)


@pytest.fixture
def test_path():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def package_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helpers():
    return Helpers

