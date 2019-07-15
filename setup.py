from setuptools import setup

setup(
   name='cosmoNODE',
   version='0.1',
   description='Neural ordinary differential equations for astrophysics',
   author='Anand Jain',
   author_email='anandj@uchicago.edu',
   packages=['cosmoNODE'],  #same as name
   install_requires=['torch', 'pandas', 'tensorflow', 'numpy'], #external packages as dependencies
)
