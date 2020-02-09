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
"""TFX BigQueryELWCExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.components.example_gen.big_query_example_gen import executor
from tfx.proto import example_gen_pb2


class BigQueryElwcExampleGen(component._QueryBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX BigQueryElwcExampleGen component.

  The BigQuery ELWC Examplegen component takes a query, and generates train
  and eval ExampleListWithContext(ELWC) examples for downstream components.
  """

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               elwc_config: example_gen_pb2.ElwcConfig = None,  # pylint: disable=redefined-builtin
               query: Optional[Text] = None,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None,
               example_artifacts: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Constructs a BigQueryExampleGen component.

    Args:
      query: BigQuery sql string, query result will be treated as a single
        split, can be overwritten by input_config.
      input_config: An example_gen_pb2.Input instance with Split.pattern as
        BigQuery sql string. If set, it overwrites the 'query' arg, and allows
        different queries per split. If any field is provided as a
        RuntimeParameter, input_config should be constructed as a dict with the
        same field names as Input proto message.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1. If any field is provided as a RuntimeParameter,
        input_config should be constructed as a dict with the same field names
        as Output proto message.

      elwc_config: An exmaple_gen_pb2.ElwcConfig i

      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      instance_name: Optional unique instance name. Necessary if multiple
        BigQueryExampleGen components are declared in the same pipeline.

    Raises:
      RuntimeError: Only one of query and input_config should be set.
    """
    if bool(query) == bool(input_config):
      raise RuntimeError('Exactly one of query and input_config should be set.')
    input_config = input_config or utils.make_default_input_config(query)
    if not elwc_config:
      raise RuntimeError('ELWC config should be set.')
    super(BigQueryElwcExampleGen, self).__init__(
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        instance_name=instance_name)
