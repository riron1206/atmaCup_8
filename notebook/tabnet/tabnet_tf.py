import os

# tfの警告出さないようにする
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # なぜか効かない
import tensorflow as tf
import tensorflow_addons as tfa

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# ===================================================================================================
# ---------------------------------------------- TabNet ---------------------------------------------
# https://www.kaggle.com/anonamename/moa-stacked-tabnet-baseline-tensorflow-2-0?scriptVersionId=45799275
# ===================================================================================================
# # Model Functions
#
# Modified from https://github.com/titu1994/tf-TabNet to support multi-label classification


def register_keras_custom_object(cls):
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    if n_units is None:
        n_units = tf.shape(x)[-1] // 2

    return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])


"""
Code replicated from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/sparsemax.py
"""


@register_keras_custom_object
@tf.function
def sparsemax(logits, axis):
    """Sparsemax activation function [1].
    For each batch `i` and class `j` we have
      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
    [1]: https://arxiv.org/abs/1602.02068
    Args:
        logits: Input tensor.
        axis: Integer, axis along which the sparsemax operation is applied.
    Returns:
        Tensor, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


"""
Code replicated from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/normalizations.py
"""


@register_keras_custom_object
class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups: int = 2,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Training=none is just for compat with batchnorm signature call
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


class TransformBlock(tf.keras.Model):
    def __init__(
        self,
        features,
        norm_type,
        momentum=0.9,
        virtual_batch_size=None,
        groups=2,
        block_name="",
        **kwargs,
    ):
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.norm_type = norm_type
        self.momentum = momentum
        self.groups = groups
        self.virtual_batch_size = virtual_batch_size

        self.transform = tf.keras.layers.Dense(
            self.features, use_bias=False, name=f"transformblock_dense_{block_name}"
        )

        if norm_type == "batch":
            self.bn = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=momentum,
                virtual_batch_size=virtual_batch_size,
                name=f"transformblock_bn_{block_name}",
            )

        else:
            self.bn = GroupNormalization(
                axis=-1, groups=self.groups, name=f"transformblock_gn_{block_name}"
            )

    def call(self, inputs, training=None):
        x = self.transform(inputs)
        x = self.bn(x, training=training)
        return x


class TabNet(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        feature_dim=64,
        output_dim=64,
        num_features=None,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=2,
        epsilon=1e-5,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:
            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.
            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.
            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.
            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.
            - Initially large learning rate is important, which should be gradually decayed until convergence.
        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'batch' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNet, self).__init__(**kwargs)

        # Input checks
        if feature_columns is not None:
            if type(feature_columns) not in (list, tuple):
                raise ValueError("`feature_columns` must be a list or a tuple.")

            if len(feature_columns) == 0:
                raise ValueError(
                    "`feature_columns` must be contain at least 1 tf.feature_column !"
                )

            if num_features is None:
                num_features = len(feature_columns)
            else:
                num_features = int(num_features)

        else:
            if num_features is None:
                raise ValueError(
                    "If `feature_columns` is None, then `num_features` cannot be None."
                )

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim < output_dim:
            raise ValueError(
                "To compute `features_for_coef`, feature_dim must be larger than output dim"
            )

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.0:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.0:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ["batch", "group"]:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        # if num_decision_steps > 1:
        # features_for_coeff = feature_dim - output_dim
        # print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(
                feature_columns, trainable=True
            )

            if self.norm_type == "batch":
                self.input_bn = tf.keras.layers.BatchNormalization(
                    axis=-1, momentum=batch_momentum, name="input_bn"
                )
            else:
                self.input_bn = GroupNormalization(
                    axis=-1, groups=self.num_groups, name="input_gn"
                )

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(
            2 * self.feature_dim,
            self.norm_type,
            self.batch_momentum,
            self.virtual_batch_size,
            self.num_groups,
            block_name="f1",
        )

        self.transform_f2 = TransformBlock(
            2 * self.feature_dim,
            self.norm_type,
            self.batch_momentum,
            self.virtual_batch_size,
            self.num_groups,
            block_name="f2",
        )

        self.transform_f3_list = [
            TransformBlock(
                2 * self.feature_dim,
                self.norm_type,
                self.batch_momentum,
                self.virtual_batch_size,
                self.num_groups,
                block_name=f"f3_{i}",
            )
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(
                2 * self.feature_dim,
                self.norm_type,
                self.batch_momentum,
                self.virtual_batch_size,
                self.num_groups,
                block_name=f"f4_{i}",
            )
            for i in range(self.num_decision_steps)
        ]

        self.transform_coef_list = [
            TransformBlock(
                self.num_features,
                self.norm_type,
                self.batch_momentum,
                self.virtual_batch_size,
                self.num_groups,
                block_name=f"coef_{i}",
            )
            for i in range(self.num_decision_steps - 1)
        ]

        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)

        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.0

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.=
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (
                glu(transform_f2, self.feature_dim) + transform_f1
            ) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (
                glu(transform_f3, self.feature_dim) + transform_f2
            ) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (
                glu(transform_f4, self.feature_dim) + transform_f3
            ) * tf.math.sqrt(0.5)

            if ni > 0 or self.num_decision_steps == 1:
                decision_out = tf.nn.relu(transform_f4[:, : self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(
                        self.num_decision_steps - 1, tf.float32
                    )

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim :]

            if ni < (self.num_decision_steps - 1):
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = self.transform_coef_list[ni](
                    features_for_coef, training=training
                )
                mask_values *= complementary_aggregated_mask_values
                mask_values = sparsemax(mask_values, axis=-1)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                    self.relaxation_factor - mask_values
                )

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1
                    )
                ) / (tf.cast(self.num_decision_steps - 1, tf.float32))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                # Visualization of the feature selection mask at decision step ni
                # tf.summary.image(
                #     "Mask for step" + str(ni),
                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                #     max_outputs=1)
                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.0

        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)

        # Visualization of the aggregated feature importances
        # tf.summary.image(
        #     "Aggregated mask",
        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        #     max_outputs=1)

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask

        return output_aggregated

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask


class TabNetClassifier(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_classes,
        num_features=None,
        feature_dim=64,
        output_dim=64,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=1,
        epsilon=1e-5,
        multi_label=False,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:
            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.
            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.
            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.
            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.
            - Initially large learning rate is important, which should be gradually decayed until convergence.
        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_classes: Number of classes.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(
            feature_columns=feature_columns,
            num_features=num_features,
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            norm_type=norm_type,
            batch_momentum=batch_momentum,
            virtual_batch_size=virtual_batch_size,
            num_groups=num_groups,
            epsilon=epsilon,
            **kwargs,
        )

        if multi_label:

            self.clf = tf.keras.layers.Dense(
                num_classes, activation="sigmoid", use_bias=False, name="classifier"
            )

        else:

            self.clf = tf.keras.layers.Dense(
                num_classes, activation="softmax", use_bias=False, name="classifier"
            )

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)


class TabNetRegressor(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_regressors,
        num_features=None,
        feature_dim=64,
        output_dim=64,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=1,
        epsilon=1e-5,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:
            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.
            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.
            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.
            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.
            - Initially large learning rate is important, which should be gradually decayed until convergence.
        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_regressors: Number of regression variables.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetRegressor, self).__init__(**kwargs)

        self.num_regressors = num_regressors

        self.tabnet = TabNet(
            feature_columns=feature_columns,
            num_features=num_features,
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            norm_type=norm_type,
            batch_momentum=batch_momentum,
            virtual_batch_size=virtual_batch_size,
            num_groups=num_groups,
            epsilon=epsilon,
            **kwargs,
        )

        self.regressor = tf.keras.layers.Dense(
            num_regressors, use_bias=False, name="regressor"
        )

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.regressor(self.activations)
        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)


# Aliases
TabNetClassification = TabNetClassifier
TabNetRegression = TabNetRegressor


class StackedTabNet(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_layers=1,
        feature_dim=64,
        output_dim=64,
        num_features=None,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=2,
        epsilon=1e-5,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.
        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:
            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.
            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.
            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.
            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.
            - Initially large learning rate is important, which should be gradually decayed until convergence.
        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_layers: Number of TabNets to stack together.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer. Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
                Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'batch' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(StackedTabNet, self).__init__(**kwargs)

        if num_layers < 1:
            raise ValueError("`num_layers` cannot be less than 1")

        if type(feature_dim) not in [list, tuple]:
            feature_dim = [feature_dim] * num_layers

        if type(output_dim) not in [list, tuple]:
            output_dim = [output_dim] * num_layers

        if len(feature_dim) != num_layers:
            raise ValueError("`feature_dim` must be a list of length `num_layers`")

        if len(output_dim) != num_layers:
            raise ValueError("`output_dim` must be a list of length `num_layers`")

        self.num_layers = num_layers

        layers = []
        layers.append(
            TabNet(
                feature_columns=feature_columns,
                num_features=num_features,
                feature_dim=feature_dim[0],
                output_dim=output_dim[0],
                num_decision_steps=num_decision_steps,
                relaxation_factor=relaxation_factor,
                sparsity_coefficient=sparsity_coefficient,
                norm_type=norm_type,
                batch_momentum=batch_momentum,
                virtual_batch_size=virtual_batch_size,
                num_groups=num_groups,
                epsilon=epsilon,
            )
        )

        for layer_idx in range(1, num_layers):
            layers.append(
                TabNet(
                    feature_columns=None,
                    num_features=output_dim[layer_idx - 1],
                    feature_dim=feature_dim[layer_idx],
                    output_dim=output_dim[layer_idx],
                    num_decision_steps=num_decision_steps,
                    relaxation_factor=relaxation_factor,
                    sparsity_coefficient=sparsity_coefficient,
                    norm_type=norm_type,
                    batch_momentum=batch_momentum,
                    virtual_batch_size=virtual_batch_size,
                    num_groups=num_groups,
                    epsilon=epsilon,
                )
            )

        self.tabnet_layers = layers

    def call(self, inputs, training=None):
        x = self.tabnet_layers[0](inputs, training=training)

        for layer_idx in range(1, self.num_layers):
            x = self.tabnet_layers[layer_idx](x, training=training)

        return x

    @property
    def tabnets(self):
        return self.tabnet_layers

    @property
    def feature_selection_masks(self):
        return [tabnet.feature_selection_masks for tabnet in self.tabnet_layers]

    @property
    def aggregate_feature_selection_mask(self):
        return [
            tabnet.aggregate_feature_selection_mask for tabnet in self.tabnet_layers
        ]


class StackedTabNetClassifier(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_classes,
        num_layers=1,
        feature_dim=64,
        output_dim=64,
        num_features=None,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=2,
        epsilon=1e-5,
        multi_label=False,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

        # Hyper Parameter Tuning (Excerpt from the paper: GUIDELINES FOR HYPERPARAMETER)
        ここでは、フィッティングの難易度を変えながら、約10Kから約10Mの学習点の範囲のデータセットを考慮しています。
        TabNetは、ハイパーパラメータに関するいくつかの一般的な原理を用いることで、すべてのデータに対して高い性能を得ることができます:
            - ほとんどのデータセットでは、Nsteps ∈[3, 10]で最良の結果が得られます。一般的に、大規模なデータセットや複雑なタスクでは、より大きなNstepsが必要となります。
              Nstepsの値が非常に高いと、オーバーフィッティングが発生し、一般化が悪くなることがあります。
            - NdとNaの値を調整することは、性能と複雑さの間のトレードオフを得るための最も効率的な方法です。
              Nd = Na はほとんどのデータセットでは妥当な選択です。
              NdとNaの値が非常に高いと、オーバーフィットが発生し、一般化が悪くなることがあります。
            - 最適なγの選択は、全体的な性能に大きな影響を与える可能性がある。一般的に、Nsteps値が大きいほど、γ値が大きい方が有利である。
            - バッチサイズを大きくすることは性能に有利です。メモリの制約が許すならば、訓練データセットの総サイズの1～10%の大きさが推奨されます。
              仮想バッチサイズは通常、バッチサイズよりもはるかに小さい。
            much smaller than the batch size.
            - 最初は大きな学習率が重要であり、収束するまで徐々に減衰させるべきである。

        # 論文のパラメの記述抜粋(APPENDIX A EXPERIMENT HYPERPARAMETERS):
        - NdとNaは{8, 16, 24, 32, 64, 128}
        - Nstepsは{3, 4, 5, 6, 7, 8, 9, 10}
        - γは{1.0, 1.2, 1.5, 2.0}
        - λspar seは{0, 0.000001, 0.0001, 0.0001, 0.001, 0.01, 0.01, 0.01. 001、0.01、0.1}
        - Bは{256、512、1024、2048、4096、8192、16384、32768}
        - BVは{256、512、1024、2048、4096}
        - mBは{0.6、0.7、0.8、0.9、0.95、0.98}
        から選択される。
        モデルサイズが所望のcuto以下でない場合は、サイズ制約を満たすように値を減少させます。
        すべての比較モデルについて、同じ探索ステップ数でハイパーパラメータチューニングを実行する。

        Args:
            feature_columns: データセットのTensorflow特徴列
            num_classes: クラス数.
            num_layers: 重ねるTabNetsの数.
            feature_dim (N_a):  feature transformation blockにおける隠れ表現の次元性。
                各層は最初に表現を2*feature_dim-dimensional出力にマッピングし、
                その半分はGLU活性化の非線形性を決定するために使用され、
                残りの半分はGLUへの入力として使用され、
                最終的にfeature_dim-dimensional出力は次の層に転送されます。
                1つの整数か、整数のリストのどちらかを指定します。
                リストの場合、レイヤー数と同じ長さでなければなりません。
            output_dim (N_d): 各decision stepの出力の次元性は，後に最終的な分類または回帰出力にマッピングされます．()
                1つの整数か、整数のリストのどちらかを指定します。
                リストの場合、レイヤー数と同じ長さでなければなりません。
            num_features: 入力特徴量の数（すなわち，各特徴量が1次元で表現されていると仮定した場合の表形式データの列数）．
            num_decision_steps(N_steps): 連続したdecision stepsの数.
            relaxation_factor (gamma): 異なる決定ステップでの各機能の再利用を促進する緩和係数。
                1の場合、ある特徴は1つの決定ステップでのみ使用されることが強制され、増加するにつれて、複数の決定ステップで特徴を使用するためのより柔軟性が提供されます。
            sparsity_coefficient (lambda_sparse): スパーシティ正則化の強さ。
                入力特徴の大部分が冗長なデータセットでは、スパース度は、より高い精度に収束するための有利な誘導バイアスを提供する可能性がある。
            norm_type: モデルに対して実行する正規化のタイプ．バッチ」または「グループ」のいずれかです。group'がデフォルトです。
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: ゴーストバッチ正規化における仮想バッチサイズ。全体のバッチサイズは virtual_batch_size の整数倍でなければなりません。
            num_groups: グループの正規化に使用したグループの数。
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(StackedTabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.stacked_tabnet = StackedTabNet(
            feature_columns=feature_columns,
            num_layers=num_layers,
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_features=num_features,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            norm_type=norm_type,
            batch_momentum=batch_momentum,
            virtual_batch_size=virtual_batch_size,
            num_groups=num_groups,
            epsilon=epsilon,
        )
        if multi_label:

            self.clf = tf.keras.layers.Dense(
                num_classes, activation="sigmoid", use_bias=False
            )

        else:

            self.clf = tf.keras.layers.Dense(
                num_classes, activation="softmax", use_bias=False
            )

    def call(self, inputs, training=None):
        self.activations = self.stacked_tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out


class StackedTabNetRegressor(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_regressors,
        num_layers=1,
        feature_dim=64,
        output_dim=64,
        num_features=None,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        norm_type="group",
        batch_momentum=0.98,
        virtual_batch_size=None,
        num_groups=2,
        epsilon=1e-5,
        **kwargs,
    ):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.
        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:
            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.
            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.
            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.
            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.
            - Initially large learning rate is important, which should be gradually decayed until convergence.
        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_regressors: Number of regressors.
            num_layers: Number of TabNets to stack together.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer. Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
                Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'batch' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(StackedTabNetRegressor, self).__init__(**kwargs)

        self.num_regressors = num_regressors

        self.stacked_tabnet = StackedTabNet(
            feature_columns=feature_columns,
            num_layers=num_layers,
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_features=num_features,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            norm_type=norm_type,
            batch_momentum=batch_momentum,
            virtual_batch_size=virtual_batch_size,
            num_groups=num_groups,
            epsilon=epsilon,
        )

        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.regressor(self.activations)
        return outl
