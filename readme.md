# How to run

1. Clone repo into Sampling in Herwig src (usually something like /src/Herwig/Sampling) 
2. Skip directly to step 6 if you use the build script
2. Create a python 3.10.16 venv. Or build from scratch
3. Install this package into venv with `pip install -e .`
4. Adjust paths in Makefile.am to link against your Python library
5. Adjust import paths in sampling.py (maybe not required depending on setup)
6. Build Sampling. If it still does not work it may be required to rebuild Herwig entirely once
7. Add the following to the top of your .in file
`
cd /Herwig/Samplers
create Herwig::PythonSampler PythonSampler
`
8. Add this somewhere before saverun
`
cd /Herwig/Samplers
set Sampler:Verbose Yes
set Sampler:BinSampler PythonSampler
cd /
`
9. Run Herwig like you do normally