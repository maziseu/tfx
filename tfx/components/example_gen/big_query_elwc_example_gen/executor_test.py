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
"""Tests for tfx.components.example_gen.big_query_elwc_example_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import apache_beam as beam
from apache_beam.testing import util
import mock
import tensorflow as tf
from tensorflow_serving.apis import input_pb2
from tfx.components.example_gen.big_query_elwc_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from google.cloud import bigquery
from google.protobuf import json_format


@beam.ptransform_fn
def _MockReadFromBigQuery(pipeline,
    query):  # pylint: disable=invalid-name, unused-argument
  mock_query_results = []

  return pipeline | beam.Create(mock_query_results)


@beam.ptransform_fn
def _MockReadFromBigQuery2(pipeline,
    query):  # pylint: disable=invalid-name, unused-argument
  mock_query_results = [
      {'qid': 1, 'feature_id_1': 1, 'feature_id_2': 1.0, 'feature_id_3': '1'},
      {'qid': 1, 'feature_id_1': 2, 'feature_id_2': 2.0, 'feature_id_3': '2'},
      {'qid': 2, 'feature_id_1': 3, 'feature_id_2': 3.0, 'feature_id_3': '3'},
      {'qid': 2, 'feature_id_1': 4, 'feature_id_2': 4.0, 'feature_id_3': '4'},
      {'qid': 3, 'feature_id_1': 5, 'feature_id_2': 5.0, 'feature_id_3': '5'}
  ]
  return pipeline | beam.Create(mock_query_results)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField('qid', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('feature_id_1', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('feature_id_2', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('feature_id_3', 'STRING', mode='REQUIRED'),
    ]
    super(ExecutorTest, self).setUp()

  @mock.patch.multiple(
      executor,
      _ReadFromBigQuery=_MockReadFromBigQuery2,
      # pylint: disable=invalid-name, unused-argument
  )
  @mock.patch.object(bigquery, 'Client')
  def testBigQueryToElwc(self, mock_client):
    # Mock query result schema for _BigQueryConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema

    with beam.Pipeline() as pipeline:
      elwc_examples = (
          pipeline | 'ToElwc' >> executor._BigQueryToElwcExample(
          elwc_config=example_gen_pb2.ElwcConfig(
              context_feature_fields=['qid']),
          input_dict={},
          exec_properties={},
          split_pattern='SELECT qid, feature_id_1, feature_id_2, feature_id_3 FROM `fake`'))

      elwc_1 = input_pb2.ExampleListWithContext()
      elwc_1.context.features.feature['qid'].int64_list.value.append(1)
      example1 = elwc_1.examples.add()
      example1.features.feature['feature_id_1'].int64_list.value.append(1)
      example1.features.feature['feature_id_2'].float_list.value.append(1.0)
      example1.features.feature['feature_id_3'].bytes_list.value.append(
        tf.compat.as_bytes('1'))
      example2 = elwc_1.examples.add()
      example2.features.feature['feature_id_1'].int64_list.value.append(2)
      example2.features.feature['feature_id_2'].float_list.value.append(2.0)
      example2.features.feature['feature_id_3'].bytes_list.value.append(
        tf.compat.as_bytes('2'))

      elwc_2 = input_pb2.ExampleListWithContext()
      elwc_2.context.features.feature['qid'].int64_list.value.append(2)
      example3 = elwc_2.examples.add()
      example3.features.feature['feature_id_1'].int64_list.value.append(3)
      example3.features.feature['feature_id_2'].float_list.value.append(3.0)
      example3.features.feature['feature_id_3'].bytes_list.value.append(
        tf.compat.as_bytes('3'))
      example4 = elwc_2.examples.add()
      example4.features.feature['feature_id_1'].int64_list.value.append(4)
      example4.features.feature['feature_id_2'].float_list.value.append(4.0)
      example4.features.feature['feature_id_3'].bytes_list.value.append(
        tf.compat.as_bytes('4'))

      elwc_3 = input_pb2.ExampleListWithContext()
      elwc_3.context.features.feature['qid'].int64_list.value.append(5)
      example5 = elwc_3.examples.add()
      example5.features.feature['feature_id_1'].int64_list.value.append(5)
      example5.features.feature['feature_id_2'].float_list.value.append(5.0)
      example5.features.feature['feature_id_3'].bytes_list.value.append(
        tf.compat.as_bytes('5'))

      expected_elwc_examples = [elwc_1, elwc_2, elwc_3]

      util.assert_that(elwc_examples, util.equal_to(expected_elwc_examples))


if __name__ == '__main__':
  tf.test.main()
