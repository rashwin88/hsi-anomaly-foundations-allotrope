# Item 10: does Miniforge 26.7.2-1 (base python 3.14) take environment.yml (python=3.12) into base?
set -x
rm -rf /tmp/mfc; bash /tmp/mf_26.7.2-1.sh -b -p /tmp/mfc > /tmp/mf_install.log 2>&1 || { tail /tmp/mf_install.log; echo MF2_FAIL install; exit; }
t=$SECONDS
/tmp/mfc/bin/mamba env update -n base -f /tmp/env10.yml > /tmp/mf_update.log 2>&1 && echo "update ok $((SECONDS-t)) s" || { tail -30 /tmp/mf_update.log; echo MF2_FAIL update; exit; }
/tmp/mfc/bin/python -c "import sys, rasterio, scipy, h5py, matplotlib; print(sys.version)"
/tmp/mfc/bin/mamba --version; /tmp/mfc/bin/conda --version
/tmp/mfc/bin/pip install -q "isofit==4.1.5" "spectral==0.25.*" > /tmp/mf_pip.log 2>&1 && /tmp/mfc/bin/python -c "import isofit; print('isofit', isofit.__version__)" || tail /tmp/mf_pip.log
echo MF2_DONE
