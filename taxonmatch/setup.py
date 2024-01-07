from setuptools import setup, find_packages

setup(
    name='taxonmatch',
    version='0.1.0',
    author='Michele Leone',
    author_email='micheleleone@outlook.com',
    description='Una breve descrizione di taxonmatch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tuo-username/taxonmatch',
    packages=find_packages(),
    install_requires=[
        # Qui puoi elencare le dipendenze necessarie, es. 'pandas', 'numpy'
    ],
    classifiers=[
        # Classificatori opzionali che aiutano gli utenti a trovare il tuo pacchetto
    ],
)