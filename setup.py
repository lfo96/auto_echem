from setuptools import setup

#setuptools.find_packages('Python_scripts')

setup(
    name='Auto EChem',
    version='0.0.1',
    packages=['auto_echem'],
    install_requires=[
        'numpy',
        'scipy',
        'galvani',
        'impedance',
        'gspread',
    ]
)
