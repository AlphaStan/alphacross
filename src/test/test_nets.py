import pytest
import numpy as np

from ..main.models import nets

@pytest.mark.parametrize("input_shape,encoding,n_players,expected_input_shape",
                         [((7, 6), '2d', 2, (7, 6)), ((7, 6), '3d', 2, (7, 6, 2))])
def test_get_input_shape_from_encoding_should_be_identity_or_add_dimension_given_valid_parameters(input_shape,
                                                                                                  encoding,
                                                                                                  n_players,
                                                                                                  expected_input_shape):
    # When
    actual_input_shape = nets._Net.get_input_shape_from_encoding(input_shape, encoding, n_players)
    # Then
    assert expected_input_shape == actual_input_shape


def test_get_input_shape_from_encoding_should_raise_ValueError_when_encoding_is_2d_and_input_shape_is_not_2d():
    with pytest.raises(ValueError):
        nets._Net.get_input_shape_from_encoding((7, 6, 2), '2d', 2)


def test_process_input_should_be_identity_when_encoding_is_2d():
    # Given
    x = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 1],
                  [1, 2, 1, 0],
                  [1, 2, 1, 1],
                  [2, 1, 2, 2]])
    expected_processed_input = x
    # When
    actual_processed_input = nets._Net.process_input(x, '2d', 2)
    # Then
    np.testing.assert_array_equal(expected_processed_input, actual_processed_input)


def test_process_input_should_do_one_hot_encoding_when_encoding_is_3d():
    # Given
    x = np.array([[[0, 0, 0, 0],
                   [0, 1, 0, 1],
                   [1, 2, 1, 0],
                   [1, 2, 1, 1],
                   [2, 1, 2, 2]]])
    expected_processed_input = np.zeros((1, 5, 4, 2))
    expected_processed_input[:, :, :, 0] = np.array([[0, 0, 0, 0],
                                                     [0, 1, 0, 1],
                                                     [1, 0, 1, 0],
                                                     [1, 0, 1, 1],
                                                     [0, 1, 0, 0]])
    expected_processed_input[:, :, :, 1] = np.array([[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [1, 0, 1, 1]])
    # When
    actual_processed_input = nets._Net.process_input(x, '3d', 2)
    # Then
    np.testing.assert_array_equal(expected_processed_input, actual_processed_input)
