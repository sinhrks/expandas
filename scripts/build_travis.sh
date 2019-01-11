if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
else
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash conda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda

#Useful for debugging any issues with conda
conda info -a

conda create -n myenv "python=$PYTHON"
source activate myenv
conda install openblas
conda install numpy scipy matplotlib "scikit-learn=$SKLEARN" "pandas=$PANDAS"
conda install patsy statsmodels seaborn

python -m pip install python-coveralls coverage
python -m pip install graphviz
python -m pip install flake8
python -m pip install xgboost
python -m pip install enum34
python -m pip install pytest pytest-cov

if [ "$IMBALANCE" ]; then
    python -m pip install imbalanced-learn
fi
