# requirements.txt in a clean Python 3.12 venv (pip only, as the Dockerfile does), then the test suite.
set -x
export MAMBA_ROOT_PREFIX=/content/mamba
/content/bin/micromamba create -y -n py312 -c conda-forge python=3.12 > /content/venv_py.log 2>&1
rm -rf /content/venv312 && /content/mamba/envs/py312/bin/python -m venv /content/venv312
V=/content/venv312/bin
$V/pip install -q --upgrade pip > /content/venv_pip.log 2>&1
t=$SECONDS
$V/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 -r /content/modT/requirements-dev.txt >> /content/venv_pip.log 2>&1 || { tail -25 /content/venv_pip.log; echo VENV_FAIL; exit 1; }
echo "pip install $((SECONDS - t)) s"
$V/python -c "import isofit, torch, rasterio; print('isofit', isofit.__version__, 'torch', torch.__version__, 'rasterio', rasterio.__version__, rasterio.__gdal_version__)"
cd /content/modT && OMP_NUM_THREADS=1 $V/python -m pytest -p no:cacheprovider 2>&1 | tail -4
echo VENV_DONE
