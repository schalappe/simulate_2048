# -*- coding: utf-8 -*-
"""
Custom optimizers
"""
import tensorflow as tf


class GCAdam(tf.keras.optimizers.Adam):
    """
    Centralized Adam optimizer.
    """
    def get_gradients(self, loss: tf.Tensor, params: list) -> list:
        """
        Returns gradients of `loss` with respect to `params`.
        Then centralized gradients.

        Parameters
        ----------
        loss: Tensor
            Loss tensor.
        params: list
            Variables

        Returns
        -------
        list
            List of gradient tensors
        """
        grads = []
        gradients = super().get_gradients(loss, params)
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads
