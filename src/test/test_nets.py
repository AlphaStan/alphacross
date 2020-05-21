import pytest
import numpy as np
import inspect
import tensorflow as tf
import os

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


@pytest.mark.parametrize("net_name",
                         [('CFDense'), ('CFDense2'), ('CFConv1'), ('CFConv2')])
def test_instantiate_nets_should_return_instance(net_name):
    # Given
    n_actions = 6
    input_shape = (7, 6)
    trainable = True
    encoding = '3d'
    n_players = 2
    # When
    net_class = getattr(inspect.getmodule(nets), net_name)
    net_instance = net_class(n_actions, input_shape, trainable, encoding, n_players)
    # Then
    assert isinstance(net_instance, net_class)


def test_nets_model_predict_method_should_return_the_same_output_given_the_same_input():
    # Given
    net = nets.CFConv2(7, (7, 6), True, '3d', 2)
    board = np.zeros((1, 7, 6, 2))
    # When
    first_pass_output = net.model.predict(board)
    second_pass_output = net.model.predict(board)
    # Then
    np.testing.assert_array_equal(first_pass_output, second_pass_output)


@pytest.mark.parametrize('net_name',
                         [net_name for net_name in os.listdir('./models/') if os.path.splitext(net_name)[1] == '.h5'])
def test_loaded_model_predict_method_should_return_the_same_output_given_the_same_input(net_name):
    # Given
    model = tf.keras.models.load_model('./models/' + net_name, custom_objects={'dqn_mask_loss': nets.dqn_mask_loss})
    board = np.random.rand(1, 7, 6)
    # When
    first_pass_output = model.predict(board)
    second_pass_output = model.predict(board)
    # Then
    np.testing.assert_array_equal(first_pass_output, second_pass_output)
