# Process and convert files from BioLogic's EC-lab https://pypi.org/project/eclabfiles/
from setuptools import setup

setup(
    name='Auto EChem',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy',
        'galvani',
        'impedance',
        'gspread'
    ]
)

# pip install 
# import scipy
# import galvani 
# print('Successfully imported galvani.')
# ## EISArt for impedance analysis https://github.com/leehangyue/EISART
# ## Authors for citation when EISArt used in scientific work.
# #import EISART.code.EISART
# #print('Successfully imported EISArt.')
