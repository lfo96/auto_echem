from setuptools import setup

setup(
    name='Auto EChem',
    version='0.0.1',
    py_modules=[Python_scripts]
    install_requires=[
        'numpy',
        'scipy',
        'galvani',
        'impedance',
        'gspread'
    ]
)
