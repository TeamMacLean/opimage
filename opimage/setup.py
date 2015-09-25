from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='opimage',
      version='0.0.1',
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
      author_email='dan.maclean@tsl.ac.uk,
      license='MIT',
      packages=['opimage'],
      install_requires=[
          'numpy',
          'skimage',
          'matplotlib'
      ],
      scripts=['lib/get_seedlings.py']
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
