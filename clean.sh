BASE_DIR="$(dirname "$(pwd)")"

cd $BASE_DIR
rm -rf Python-3.10.16.tar.xz herwig-bootstrap.py activate_herwig venv_herwig Python-3.10.16 automake automake-1.16.5
rm $BASE_DIR/Herwig/src/*_done
# redownloading the archives takes long. also sometimes the mirrors dont work so dont delete herwigs download cache
find $BASE_DIR/Herwig/src/ -mindepth 1 ! \( -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.bz2" \) -delete
