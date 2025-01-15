from setuptools import setup

setup(name='pyblinkers',
      version='0.0.6',
      description='make life easier',
      author='rpb',
      packages=['pyblinkers','pyblinkers.utilities',
                  'pyblinkers.vislab','pyblinkers.viz'],
      long_description='Get EOG from EEG signal recording. In this 0.0.6 version, we change the name from eeg-blinks to pyblinker',
      install_requires=['seaborn','mne']
      )
