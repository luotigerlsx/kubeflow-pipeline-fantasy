# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data processing and training functions for Covertype TFX pipeline."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.executor import TrainerFnArgs

NUMERIC_FEATURE_KEYS = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area', 'Soil_Type']

LABEL_KEY = 'Cover_Type'
NUM_CLASSES = 7

EXPORTED_MODEL_NAME = 'covertype-classifier'


### Helper functions used by the preprocessing_fn and trainer_fn


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor."""

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              feature_spec)

        transformed_features = model.tft_layer(parsed_features)
        transformed_features.pop(_transformed_name(LABEL_KEY))

        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(filenames, feature_specs, label_key, batch_size=200):
    """Generates features and labels for training or evaluation."""

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=filenames,
        batch_size=batch_size,
        features=feature_specs,
        label_key=label_key,
        reader=_gzip_reader_fn)

    return dataset


def _example_serving_receiver_fn(tf_transform_output, schema, label_key):
    """Build the serving graph."""

    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop(label_key)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
    """Build a simple keras wide and deep model.
    Args:
      wide_columns: Feature columns wrapped in indicator_column for wide (linear)
        part of the model.
      deep_columns: Feature columns for deep part of the model.
      dnn_hidden_units: [int], the layer sizes of the hidden DNN.
    Returns:
      A Wide and Deep Keras model
    """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    # Keras needs the feature definitions at compile time.
    # TODO(b/139081439): Automate generation of input layers from FeatureColumn.
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in _transformed_names(NUMERIC_FEATURE_KEYS)
    }
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(CATEGORICAL_FEATURE_KEYS)
    })

    # TODO(b/144500510): SparseFeatures for feature columns + Keras.
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    concat = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(model.summary())
    return model


def _build_keras_model(numeric_feature_keys,
                       categorical_feature_keys,
                       hidden_units) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.
    Args:
      hidden_units: [int], the layer sizes of the DNN (input layer first).
    Returns:
      A keras Model.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(numeric_feature_keys)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=num_buckets, default_value=0)
        for key, num_buckets in categorical_feature_keys
    ]
    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        # TODO(b/139668410) replace with premade wide_and_deep keras model
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25])
    return model


### Preprocessing function for the Transform component
def preprocessing_fn(inputs):
    """Preprocesses Covertype Dataset."""

    outputs = {}

    # Scale numerical features
    for key in NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    # Generate vocabularies and maps categorical features
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x=_fill_in_missing(inputs[key]), num_oov_buckets=1,
            vocab_filename=key)

    # Convert Cover_Type from 1-7 to 0-6
    outputs[_transformed_name(LABEL_KEY)] = _fill_in_missing(
        inputs[LABEL_KEY]) - 1

    return outputs


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(
            categorical_feature_keys=CATEGORICAL_FEATURE_KEYS,
            numeric_feature_keys=NUMERIC_FEATURE_KEYS,
            # Construct layers sizes with exponetial decay
            hidden_units=[
                max(2, int(first_dnn_layer_size * dnn_decay_factor ** i))
                for i in range(num_dnn_layers)
            ])

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output)
                .get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf',
               signatures=signatures)
