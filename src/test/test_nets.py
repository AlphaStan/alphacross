import inspect
import numpy as np
import pytest
import tensorflow as tf
import os
import json

from ..main.models import nets


@pytest.mark.parametrize("input_shape,encoding,n_players,expected_input_shape",
                         [((7, 6), '2d', 2, (7, 6)), ((7, 6), '3d', 2, (7, 6, 2)), ((7, 6, 2), '3d', 2, (7, 6, 2))])
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


def test_get_input_shape_from_encoding_should_raise_ValueError_when_encoding_is_3d_and_input_shape_is_neither_2d_nor_3d():
    with pytest.raises(ValueError):
        nets._Net.get_input_shape_from_encoding((6, 7, 6, 2), '3d', 2)


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


def test_loaded_model_predict_method_should_return_the_same_output_given_the_same_input():
    # Given
    np.random.seed(42)
    model = tf.keras.models.load_model('./src/test/resources/trained_model_15122019_234912.h5',
                                       custom_objects={'dqn_mask_loss': nets.dqn_mask_loss})
    board = np.ones((1, 7, 6))
    expected_probabilities = np.array([[3.799e-9, 2.332e-9, 9.994e-1, 4.838e-5, 5.968e-4, 4.185e-8, 1.904e-8]])
    # When)
    first_pass_output = model.predict(board)
    second_pass_output = model.predict(board)
    # Then
    np.testing.assert_array_almost_equal(first_pass_output, expected_probabilities, decimal=3)
    np.testing.assert_array_almost_equal(second_pass_output, expected_probabilities, decimal=3)


def test_save_method_should_save_the_net_attributes_and_the_keras_model(tmpdir):
    # Given
    n_actions = 7
    input_shape = [7, 6, 2]
    trainable = True
    encoding = '3d'
    n_players = 2
    net_name = 'CFConv2'
    net = nets.CFConv2(n_actions, input_shape, trainable, encoding, n_players)
    expected_attributes = {'n_actions': n_actions,
                           'input_shape': input_shape,
                           'trainable': trainable,
                           'encoding': encoding,
                           'n_players': n_players,
                           'net_name': net_name}
    # When
    net.save(tmpdir)
    with open(os.path.join(tmpdir, 'attributes.json')) as data:
        actual_attributes = json.load(data)
    # Then
    assert os.path.exists(os.path.join(tmpdir, 'model.h5'))
    assert actual_attributes == expected_attributes


def test_load_model_should_return_an_instance_of_net():
    # Given
    load_dir = 'src/test/resources'
    with open(os.path.join(load_dir, 'attributes.json')) as data:
        expected_attributes = json.load(data)
    expected_loaded_attributes = {key: expected_attributes[key] for key in expected_attributes if key != 'net_name'}
    expected_net_class = getattr(inspect.getmodule(nets), expected_attributes['net_name'])
    expected_model = tf.keras.models.load_model(os.path.join(load_dir, 'model.h5'),
                                                custom_objects={'dqn_mask_loss': nets.dqn_mask_loss})
    # When
    loaded_net = nets.load_net(load_dir)
    actual_loaded_attributes = {key: loaded_net.__dict__[key] for key in loaded_net.__dict__
                         if key not in ['model', 'net_name']}
    actual_model = loaded_net.model
    # Then
    assert isinstance(loaded_net, expected_net_class)
    assert actual_loaded_attributes == expected_loaded_attributes
    for actual_weights, expected_weights in zip(actual_model.get_weights(), expected_model.get_weights()):
        np.testing.assert_array_equal(actual_weights, expected_weights)
