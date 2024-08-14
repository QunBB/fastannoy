from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import codecs
import os
import platform
import sys

_version_ = '1.1.0'

readme_note = """\

   GitHub repository <https://github.com/QunBB/fastannoy>

"""

with codecs.open('README.md', encoding='utf-8') as fobj:
    long_description = readme_note + fobj.read()

# Various platform-dependent extras
extra_compile_args = ['-D_CRT_SECURE_NO_WARNINGS']
extra_link_args = []

# Not all CPUs have march as a tuning parameter
cputune = ['-march=native',]
if platform.machine() == 'ppc64le':
    extra_compile_args += ['-mcpu=native',]

if platform.machine() == 'x86_64':
    extra_compile_args += cputune

if os.name != 'nt':
    extra_compile_args += ['-O3', '-ffast-math', '-fno-associative-math']

# Add multithreaded build flag for all platforms using Python 3 and
# for non-Windows Python 2 platforms
python_major_version = sys.version_info[0]
if python_major_version == 3 or (python_major_version == 2 and os.name != 'nt'):
    extra_compile_args += ['-DANNOYLIB_MULTITHREADED_BUILD']

    if os.name != 'nt':
        extra_compile_args += ['-std=c++14']

# #349: something with OS X Mojave causes libstd not to be found
if platform.system() == 'Darwin':
    extra_compile_args += ['-mmacosx-version-min=10.12']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.12']

# Manual configuration, you're on your own here.
manual_compiler_args = os.environ.get('ANNOY_COMPILER_ARGS', None)
if manual_compiler_args:
    extra_compile_args = manual_compiler_args.split(',')
manual_linker_args = os.environ.get('ANNOY_LINKER_ARGS', None)
if manual_linker_args:
    extra_link_args = manual_linker_args.split(',')

setup(name='fastannoy',
      version=_version_,
      description='Faster version of Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['fastannoy'],
      package_data={'fastannoy': ['__init__.pyi', 'py.typed']},
      ext_modules=[
          Pybind11Extension(
              'fastannoy.annoylib', ['src/annoymodule.cpp'],
              language='c++',
              depends=['src/annoylib.h', 'src/kissrandom.h', 'src/mman.h'],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              define_macros=[("VERSION_INFO", _version_)]
          )
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Qun',
      author_email='myqun20190810@163.com',
      url='https://github.com/QunBB/fastannoy',
      license='Apache License 2.0',
      python_requires=">=2.7",
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
      ],
      keywords='nns, approximate nearest neighbor search',
      setup_requires=['nose>=1.0', 'pybind11']
      )
