from setuptools import setup

#setuptools.find_packages('Python_scripts')

setup(
    name='Auto EChem',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy',
        'galvani',
        'impedance',
        'gspread',
        py_modules = [
            'Python_scripts\auto',
            'Python_scripts\GCPL_functions',
            'Python_scripts\impedance_functions',
            'Python_scripts\general_functions',
            'Python_scripts\TGA_DSC_functions',
            'Python_scripts\three_el_functions',
            'Python_scripts\XPS_functions',
            ],
    ]
)
