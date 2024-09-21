import sys
if sys.version_info.major < 3 and sys.version_info.micro < 15:
    from distribute_setup import use_setuptools
    use_setuptools()

with open("README.md", "r") as fh:
    long_description = fh.read()
    
from setuptools import setup, find_packages

setup(name             = 'rogues',
      version          = '1.0.0',
      test_suite       = 'nose.collector',
      packages         = find_packages(),
      install_requires = ['numpy', 'scipy'],
      author           = 'Don MacMillen',
      author_email     = 'don@macmillen.net',
      url              = 'https://github.com/macd/rogues',
      description      = "Python and numpy port of Nicholas Higham's m*lab test matrices",
      long_description = long_description,
      long_description_content_type = "text/markdown",
      license          = 'MIT',
      keywords         = 'numpy scipy matplotlib linalg',
      zip_safe         = True,
      classifiers=[
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: MIT License',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering :: Mathematics',
      ],
)
