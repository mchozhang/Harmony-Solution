#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tf_agents.networks import network, q_network


class MaskedQNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 mask_q_value=-100000,
                 fc_layer_params=(75, 40),
                 activation_fn=tf.keras.activations.relu,
                 name='MaskedQNetwork'):
        super(MaskedQNetwork, self).__init__(input_tensor_spec,
                                             state_spec=(),
                                             name=name)
        self._q_net = q_network.QNetwork(input_tensor_spec['state'],
                                         action_spec=action_spec,
                                         fc_layer_params=fc_layer_params,
                                         activation_fn=activation_fn)

        self._mask_q_value = mask_q_value

    def call(self, observations, step_type=None, network_state=()):
        state = observations['state']
        mask = observations['mask']

        q_values, _ = self._q_net(state, step_type)
        small_constant = tf.constant(self._mask_q_value,
                                     dtype=q_values.dtype,
                                     shape=q_values.shape)
        zeros = tf.zeros(shape=mask.shape, dtype=mask.dtype)
        masked_q_values = tf.where(tf.math.equal(zeros, mask),
                                   small_constant, q_values)
        return masked_q_values, network_state
