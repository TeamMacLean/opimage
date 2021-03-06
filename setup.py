from setuptools import setup

def readme():
    with open('README.md') as readme:
        return readme.read()

setup(name='opimage',
      version='0.0.1dev',
      description='Open PI Image Library',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Image Processing :: Science',
      ],
      keywords='plant image processing',
      url='http://github.com/danmaclean/opimage',
      author='Team MacLean',
      author_email='dan.maclean@tsl.ac.uk',
      license='MIT',
      packages=['opimage'] ,
      install_requires=[
          'numpy',
          'scipy',
          'scikit-image',
          'matplotlib',
          'python-dateutil',
          'schedule'
      ],
      scripts=['opimage/scripts/get_seedlings.py'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
