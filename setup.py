from setuptools import setup, find_packages

setup(
    name='HiDDeN-tensorflow',
    url='https://github.com/marco-willi/HiDDeN-tensorflow',
    author='Marco Willi',
    version='1.0',
    packages=find_packages(),
    package_data={'': ['*.bmp', '*.jpeg']},
    include_package_data=True,
    install_requires=[
        'tensorflow>=2.0',
        'tensorflow-probability>=0.7.0',
        'matplotlib>=3.1.1',
        'scikit-learn>=0.21.3'
    ],
    python_requires='>=3.7'
)
