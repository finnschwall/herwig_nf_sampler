#dependencies: list may not be complete
#for building herwig
#sudo apt-get install autoconf cmake
#if on pc without root access (cluster?) you can build both on your own. Dont forget to then adjust LD_LIBRARY_PATH

#for building python
#zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel libffi-devel xz-devel

cd ..
export CORES=4
BASE_DIR="$(dirname "$(pwd)")"


HERWIG_HOME="$BASE_DIR/Herwig"
PYTHON_BASE="$BASE_DIR/Python-3.10.16"
PYTHON_INSTALL="$BASE_DIR/Python-3.10.16/install"
GIT_REPO_DIR="$BASE_DIR/herwig_nf_sampling"

mkdir -p "$HERWIG_HOME"
mkdir -p "$PYTHON_INSTALL"
mkdir -p "$PYTHON_BASE"

export HERWIG_HOME
export PYTHON_INSTALL


#download and build python. makes sure we dont get any path issues with system installation and works without root
wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tar.xz
tar -xf Python-3.10.16.tar.xz
cd Python-3.10.16/
./configure --prefix=$PYTHON_INSTALL \
            --enable-optimizations \
            --with-ensurepip=install \
            --enable-shared
make -j${CORES} && make install

$PYTHON_INSTALL/bin/python3 -m venv venv_herwig


#set paths for python and for proper linking against our new python installation
export PATH="$PYTHON_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$PYTHON_INSTALL/lib:$LD_LIBRARY_PATH"
export PYTHONHOME="$PYTHON_INSTALL"
export PYTHONPATH="$PYTHON_INSTALL/lib/python3.10"
export LD_LIBRARY_PATH="$PYTHON_INSTALL/lib:$LD_LIBRARY_PATH"


source venv_herwig/bin/activate 

#install python packages that we need for linking/can cause issues
pip3 install --upgrade pip
pip3 install cython pybind11 torch torchtestcase
#install python packages that are needed later (but should not cause any issues)
pip3 install numpy tqdm matplotlib decouple scipy

#check if torch.cuda is available
cuda_available=\$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$cuda_available" != "True" ]; then
  echo "Pytorch was installed without CUDA support. Check your CUDA installation. Without you will only be able to use this library for inference!"
fi

#build herwig. See https://herwig.hepforge.org/tutorials/installation/bootstrap.html
#lean back and enjoy. this may take a while
#if on cluster crank up the -j
python3 herwig-bootstrap.py --lite -j ${CORES} ${HERWIG_HOME}

cp -r $GIT_REPO_DIR/* "$HERWIG_HOME/Herwig/src/Herwig-7.3.0/Sampling"


cd "$HERWIG_HOME/Herwig/src/Herwig-7.3.0/Sampling"
#NEED TO CHANGE PATHS IN Makefile.am


make -j ${CORES} && make install

cd "$BASE_DIR"

cat << EOL > "activate_herwig.sh"
BASE_DIR=$BASE_DIR
HERWIG_HOME="$BASE_DIR/Herwig"
export PATH="$PYTHON_INSTALL/bin:\$PATH"
export LD_LIBRARY_PATH="$PYTHON_INSTALL/lib:\$LD_LIBRARY_PATH"
export PYTHONHOME="$PYTHON_INSTALL"
export PYTHONPATH="$PYTHON_INSTALL/lib/python3.10"
source \$HERWIG_HOME/bin/activate
EOL

echo "Finished!"
echo "To activate the environment, run:"
echo "source $BASE_DIR/activate_herwig.sh"
if [ "$cuda_available" != "True" ]; then
    echo "REMINDER: Pytorch was installed without CUDA support. Check your CUDA installation. Without you will only be able to use this library for inference!"
fi