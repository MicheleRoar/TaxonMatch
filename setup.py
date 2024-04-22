from setuptools import setup, find_packages
from pip._internal.req import parse_requirements


requirements = parse_requirements('requirements.txt', session='hack')
required = [str(req.requirement) for req in requirements]

setup(
    name='TaxonMatch',
    version='0.1.0',
    author='Michele Leone',
    author_email='micheleleone@outlook.com',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MicheleRoar/TaxonMatch',
    packages=find_packages(),
    include_package_data=True
    install_requires=required,  
)
