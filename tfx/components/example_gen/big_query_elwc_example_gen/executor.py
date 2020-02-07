# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic TFX BigQueryElwcExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Dict, List, Text

import apache_beam as beam
import tensorflow as tf
import tensorflow_serving

from google.cloud import bigquery
from tfx import types
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from tensorflow_serving.apis import input_pb2


class _BigQueryElwcConverter(object):
  """Help class for bigquery result row to ExampleListWithContext example conversion."""

  def __init__(self, elwc_config: example_gen_pb2.ElwcConfig, query: Text):
    client = bigquery.Client()
    # Dummy query to get the type information for each field.
    query_job = client.query('SELECT * FROM ({}) LIMIT 0'.format(query))
    results = query_job.result()
    self._type_map = {}
    self._context_feature_fields = set(elwc_config.context_feature_fields)
    self.ContextFeature = collections.namedtuple('ContextFeature', self._context_feature_fields)
    field_names = set()
    for field in results.schema:
      self._type_map[field.name] = field.field_type
      field_names.add(field.name)
    # Check whether the query contains the necessary context fields.
    if not field_names.issuperset(self._context_feature_fields):
        raise RuntimeError('Some context feature fields are missing from the query')

  def RowToContextFeature(self, instance: Dict[Text, Any]) -> collections.namedtuple:
    context_data = instance.fromkeys(self._context_feature_fields)
    return self.ContextFeature(**context_data)

  def RowToExampleWithoutContext(self, instance: Dict[Text, Any]) -> tf.train.Example:
    """Convert bigquery result row to tf example."""
    example_data = instance.fromkeys(instance.keys() - self._context_feature_fields)
    example_feature = self.DataToFeatures(example_data)
    return tf.train.Example(features=example_feature)

  def DataToFeatures(self, instance: Dict[Text, Any]) -> tf.train.Features:
    feature = {}
    for key, value in instance.items():
      data_type = self._type_map[key]
      if value is None:
        feature[key] = tf.train.Feature()
      elif data_type == 'INTEGER':
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
      elif data_type == 'FLOAT':
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value]))
      elif data_type == 'STRING':
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
      else:
        # TODO(jyzhao): support more types.
        raise RuntimeError(
            'BigQuery column type {} is not supported.'.format(data_type))
    return tf.train.Features(feature=feature)


  def CombineContextAndExampleFeatures(self,
                                       context_feature: collections.namedtuple,
                                       example_features: List[tf.train.Example])-> input_pb2.ExampleListWithContext:
    context_feature_dict = context_feature.toDict()
    context_feature = self.DataToFeatures(context_feature_dict)
    return input_pb2.ExampleListWithContext(
      context=context_feature,
      examples=example_features)


# Create this instead of inline in _BigQueryToExample for test mocking purpose.
@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.typehints.Dict[Text, Any])
def _ReadFromBigQuery(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, query: Text) -> beam.pvalue.PCollection:
  return (pipeline
          | 'QueryTable' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True)))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(input_pb2.ExampleListWithContext)
def _BigQueryToElwcExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.Artifact]],  # pylint: disable=unused-argument
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read from BigQuery and transform to ELWC.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a BigQuery sql string.

  Returns:
    PCollection of TF examples.
  """
  converter = _BigQueryElwcConverter(split_pattern)

  return (pipeline
          | 'QueryTable' >> _ReadFromBigQuery(split_pattern)  # pylint: disable=no-value-for-parameter
          | 'SeparateContextAndFeature' >> beam.Map(lambda row: (converter.RowToContext(row), converter.RowToExampleFeatures(row)))
          | 'GroupByContext' >> beam.GroupByKey()
          | 'ToTFExample' >> beam.Map(converter.combineContextAndExampleFeatures))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for BigQuery to TF examples."""
    return _BigQueryToElwcExample
