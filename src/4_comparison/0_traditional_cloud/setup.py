#!/usr/bin/env python
# Copyright 2019 Google LLC
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
# ==============================================================================
#
# from setuptools import find_packages
# from setuptools import setup
#
# REQUIRED_PACKAGES = [
#     'scikit-learn>=0.20.2',
#     'pandas==0.24.2',
#     'cloudml-hypertune',
# ]
#
# setup(
#     name='trainer',
#     version='0.1',
#     install_requires=REQUIRED_PACKAGES,
#     packages=find_packages(),
#     include_package_data=True,
#     description='AI Platform | Training | scikit-learn | Base'
# )

"""Config for installing a python module/package."""

from setuptools import setup, find_packages

NAME = 'MTLCC_MODIS'
AUTHOR = 'Alejandro Coca-Castro',
EMAIL = 'acocac@gmail.com',
VERSION = '0.1'
REQUIRED_PACKAGES = ['configparser','cloudml-hypertune','scikit-learn>=0.20.2','pandas==0.24.2']
LICENSE = 'MIT'
DESCRIPTION = 'Run scikit-learn in GCP'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False)
