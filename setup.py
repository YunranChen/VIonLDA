from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='UTF-8') as f:
    long_description = f.read()

setup(
    name='VIonLDA',  
    version='2.4', 
    description='Variational Inference on LDA model. Reproduce Blei et al., 2003.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    url='https://github.com/YunranChen/VIonLDA',  # Optional
    author='YunranChen,JunwenHuang', 
    author_email='yunran.chen@duke.edu',  # Optional
    #py_modules = ['VIonLDA'],
    #scripts = ['lda.py'],
    packages=find_packages(exclude=['tests','Examples','parallel']),  # Required
    #package_data={  # Optional
    #    'VIonLDA': ['ap.txt'],
    #}
    python_requires='>=3',
    
)
