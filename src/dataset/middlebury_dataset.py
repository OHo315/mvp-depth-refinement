# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
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
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import torch

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
from .base_normals_dataset import BaseNormalsDataset


class MiddleburyDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Middlebury dataset parameter
            # NOTE: These values are copied from nyu. 
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=False,  # our depths are raw and do not have filled in holes as a result of sensor measurements, like in NYU.
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Convert depth from mm to m.
        depth_decoded = depth_in / 1000.0
        return depth_decoded


class MiddleburyNormalsDataset(BaseNormalsDataset):
    pass
