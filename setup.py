"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import setuptools

base_requires = [
    'omegaconf==2.3.0',
    'transformers>=4.26.0,<5.0.0',
    'torch>=2.0.0,<3.0.0',
    'torchmetrics>=0.9.3,<1.0.0',
    'torchvision>=0.15.0,<1.0.0',
    'pytz>=2021.3,<2024.0',
    'pandas>=1.5.0,<2.0.0',
    'scikit-learn>=1.3.2,<2.0.0',
]

setuptools.setup(
    name='lanistr',
    version='0.1.0',
    install_requires=base_requires,
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
