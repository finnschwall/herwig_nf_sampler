# How to run

1. Clone repo into a folder where you want your install (e.g. so that you have Herwig/herwig_nf_sampler)
2. Adjust the number of cores for building inside install.sh (don't forget this if you are on a cluster)
3. Execute install.sh
4. Reflect on life for 1-4 hours
5. Check that the script really finished with all checks
7. Add the following to the top of your .in file
```
cd /Herwig/Samplers
create Herwig::PythonSampler PythonSampler
```
8. Add this somewhere before saverun
```
cd /Herwig/Samplers
set Sampler:Verbose Yes
set Sampler:BinSampler PythonSampler
cd /
```
9. Run Herwig like you do normally

10. Do not rerun the install script if you have an existing install. Use the newly created activate