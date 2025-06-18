#!/bin/bash
#for building python
#zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel libffi-devel xz-devel

export CORES=32
BASE_DIR="$(dirname "$(pwd)")"

HERWIG_HOME="$BASE_DIR/Herwig"
PYTHON_BASE="$BASE_DIR/Python-3.10.16"
PYTHON_INSTALL="$BASE_DIR/Python-3.10.16/install"
GIT_REPO_DIR="$BASE_DIR/herwig_nf_sampler"
SAMPLING="$HERWIG_HOME/src/Herwig-7.3.0/Sampling"

mkdir -p "$HERWIG_HOME"
mkdir -p "$PYTHON_INSTALL"
mkdir -p "$PYTHON_BASE"

export HERWIG_HOME
export PYTHON_INSTALL

python_exists="false"

cd $BASE_DIR



if [ -f "${PYTHON_INSTALL}/bin/python3" ]; then
	echo "Python ${PYTHON_VERSION} is already installed at ${PYTHON_INSTALL}"
	python_exists="true"
else
	wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tar.xz
	tar -xf Python-3.10.16.tar.xz
	cd Python-3.10.16/
	./configure \
		--enable-shared \
		--enable-loadable-sqlite-extensions \
		--with-readline \
		--enable-optimizations \
		--with-ensurepip=install \
		--enable-ipv6 \
		--disable-test-suite \
		--prefix=${PYTHON_INSTALL} \
		&& make -j ${nCores} \
		&& make install 
	ln -s ${PYTHON_INSTALL}/lib64/python3.10/lib-dynload/ ${PYTHON_INSTALL}/lib/python3.10/lib-dynload	
fi

export LD_LIBRARY_PATH="${PYTHON_INSTALL}/lib64"

if [ ! -f "$BASE_DIR/venv_herwig/bin/python3" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_INSTALL/bin/python3 -m venv "$BASE_DIR/venv_herwig"
    
    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully"
    else
        echo "Failed to create virtual environment"
        exit 1
    fi
else
    echo "Virtual environment already exists"
fi

. $BASE_DIR/venv_herwig/bin/activate 

if [ "$python_exists" = "true" ]; then
	#install python packages that we need for linking/can cause issues
	pip3 install --upgrade pip
	pip3 install cython pybind11 torch torchtestcase install numpy tqdm matplotlib python-decouple scipy six
	#install python packages that are needed later (but should not cause any issues)
	#pip3 install numpy tqdm matplotlib python-decouple scipy
else
    echo "Assuming pip packages are already installed as python was already built"
fi

#check if torch.cuda is available
cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$cuda_available" != "True" ]; then
  echo "Pytorch was installed without CUDA support. Check your CUDA installation. Without you will only be able to use this library for inference!"
fi


# build correct automake version for herwig
if [ ! -f "$BASE_DIR/automake-1.16.5/install/bin/automake" ]; then
	wget https://ftp.gnu.org/gnu/automake/automake-1.16.5.tar.xz
	tar -xf automake-1.16.5.tar.xz
	cd automake-1.16.5
	./configure --prefix=$BASE_DIR/automake-1.16.5/install
	make -j ${nCores} 
	make install
fi
export PATH="$BASE_DIR/automake-1.16.5/install/bin:$PATH"

#build herwig. See https://herwig.hepforge.org/tutorials/installation/bootstrap.html
#lean back and enjoy. this may take a while
if [ ! -f "$BASE_DIR/herwig-bootstrap.py" ]; then
	echo "hello"
	wget -O $BASE_DIR/herwig-bootstrap.py https://herwig.hepforge.org/downloads/herwig-bootstrap
fi

if [ ! -f "$BASE_DIR/Herwig/bin/activate" ]; then
	python3 $BASE_DIR/herwig-bootstrap.py --lite -j ${CORES} ${HERWIG_HOME}
else
	echo "Herwig already installed"
fi


if [ ! -d "$HERWIG_HOME/src/Herwig-7.3.0/SamplingBackup" ]; then
	cp -r "$HERWIG_HOME/src/Herwig-7.3.0/Sampling" "$HERWIG_HOME/src/Herwig-7.3.0/SamplingBackup"
	echo "Backup of Sampling has been created in $HERWIG_HOME/src/Herwig-7.3.0/SamplingBackup"
else
	echo "Sampling backup already exists. Continuing will override files inside current sampling folder without any backup"
	echo "Are you sure you want to proceed? (y/n) "
	read REPLY

	case "$REPLY" in
	  [Yy]) 
	    ;;
	  *)
	    echo "Operation cancelled."
	    exit 0
	    ;;
	esac
fi

cp -a $GIT_REPO_DIR/* "$HERWIG_HOME/src/Herwig-7.3.0/Sampling"


MAKEFILE="$SAMPLING/Makefile.am"
if [ ! -f "$MAKEFILE" ]; then
    echo "Error: Makefile.am not found at $MAKEFILE. This really shouldnt happen?"
    exit 1
fi
sed -i "s|-L/mnt/data-slow/Herwig/Python-3.10.16/install/lib|-L${PYTHON_INSTALL}/lib|g" "$MAKEFILE"
sed -i "s|-I/mnt/data-slow/Herwig/Python-3.10.16/install/include/python3.10|-I${PYTHON_INSTALL}/include/python3.10|g" "$MAKEFILE"
sed -i "s|-I/mnt/data-slow/Herwig/venv_herwig/lib/python3.10/site-packages/pybind11/include|-I${BASE_DIR}/venv_herwig/lib/python3.10/site-packages/|g" "$MAKEFILE"

cd $SAMPLING

make -j ${CORES} && make install

cd "$BASE_DIR"

cat << EOL > "activate_herwig"
BASE_DIR=$BASE_DIR
HERWIG_HOME="$BASE_DIR/Herwig"
PYTHON_INSTALL="$BASE_DIR/Python-3.10.16/install"
export LD_LIBRARY_PATH="${PYTHON_INSTALL}/lib64:${PYTHON_INSTALL}/lib:$LD_LIBRARY_PATH"
export PATH="$BASE_DIR/automake-1.16.5/install/bin:$PATH"
source \$HERWIG_HOME/bin/activate
EOL

echo "\n\n\n"
echo "Finished!"
echo "To activate the environment (for running and developing), run:"
echo "source $BASE_DIR/activate_herwig"
if [ "$cuda_available" != "True" ]; then
    echo "REMINDER: Pytorch was installed without CUDA support. Check your CUDA installation. Without you will only be able to use this library for inference!"
fi