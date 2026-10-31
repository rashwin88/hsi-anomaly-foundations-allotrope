#!/usr/bin/env bash
# Item 11 runtime on colab-atm (L4): the Dockerfile builder stage replayed step for step (environment.yml in a
# micromamba env, CUDA torch, ISOFIT 4.1.5, 6S + fast builds + sixs_check + wrapper, isofit_patches, sRTMnet / data /
# examples, surface prior, empirical_line patch), then the runtime checks. Module from /content/modH.tgz.
set -x
M=/content/modH; rm -rf $M && mkdir -p $M && tar xzf /content/modH.tgz -C $M
apt-get install -y -qq gfortran > /dev/null 2>&1; gfortran --version | head -1
export MAMBA_ROOT_PREFIX=/content/mamba
mkdir -p /content/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content bin/micromamba
sed 's/^name: base/name: atm/' $M/environment.yml > /tmp/env_atm.yml
/content/bin/micromamba create -y -f /tmp/env_atm.yml > /content/setup_env.log 2>&1 || { tail -30 /content/setup_env.log; echo SETUP_FAIL env; exit 1; }
E=/content/mamba/envs/atm; export PATH=$E/bin:$PATH
pip install --no-cache-dir "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 > /content/setup_pip.log 2>&1 &&
pip install --no-cache-dir "isofit==4.1.5" "spectral==0.25.*" "pystac==1.10.*" "lxml==6.*" >> /content/setup_pip.log 2>&1 || { tail -20 /content/setup_pip.log; echo SETUP_FAIL pip; exit 1; }
isofit download sixs --path /root/.isofit/sixs --overwrite --debug-make > /content/setup_sixs.log 2>&1; ls -l /root/.isofit/sixs/sixsV* || { echo SETUP_FAIL sixs; exit 1; }
rm -rf /opt/build /opt/sixs; cp -r $M/build /opt/build
bash /opt/build/build_sixs.sh /root/.isofit/sixs /opt/sixs &&
python /opt/build/sixs_check.py /root/.isofit/sixs/sixsV2.1 /opt/sixs /opt/build/sixs_fixtures &&
rm -rf /opt/sixs/check && mkdir -p /opt/sixs/stock && mv /root/.isofit/sixs/sixsV2.1 /opt/sixs/stock/sixsV2.1 &&
install -m 755 /opt/build/sixs_wrapper.sh /root/.isofit/sixs/sixsV2.1 &&
python /opt/build/isofit_patches.py && /root/.isofit/sixs/sixsV2.1 --selftest || { echo SETUP_FAIL 6S; exit 1; }
isofit download srtmnet --path /root/.isofit/srtmnet > /content/setup_dl.log 2>&1
isofit download data --path /root/.isofit/data >> /content/setup_dl.log 2>&1
isofit download examples --path /root/.isofit/examples >> /content/setup_dl.log 2>&1
mkdir -p /opt/aux/surface; ln -sfn /root/.isofit/data /opt/aux/isofit-data
EL=$E/lib/python3.12/site-packages/isofit/utils/empirical_line.py
sed -i "s/\(output_reflectance_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/; s/\(output_uncertainty_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/" "$EL"; grep -c 'n_input_bands, nls' "$EL"
export PYTHONPATH=$M/src:$M
for b in /opt/sixs/*/sixsV2.1; do ldd "$b" | grep "not found"; done
python -c "import importlib, fast_al; A = importlib.import_module('isofit.utils.apply_oe'); assert A.ALAlg is fast_al.analytical_line; print('apply_oe uses fast_al')"
python -c "import isofit, torch, rasterio, spectral, h5py, matplotlib, scipy, pyproj; import readers.prep, atm_post, harmonise, writers, report; print('isofit', isofit.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available()); from osgeo import gdal; print('gdal', gdal.__version__)"
ls -lh /root/.isofit/srtmnet
echo SETUP_DONE
