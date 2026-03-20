# Setting Up Remote VM

## VM Details

| Field      | Value              |
|------------|--------------------|
| Name       | ntro--vm57         |
| IP Address | 103.42.50.245      |
| Port       | 2271               |
| UserID     | user57             |
| OS         | Ubuntu 24.04.2 LTS |
| Arch       | x86_64             |
| Pkg Mgr    | apt                |

## 1. Connect

```bash
ssh -p 2271 user57@103.42.50.245
```

## 2. Install Essentials

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git curl wget unzip tmux screen
```

## 3. Verify

```bash
python3 --version && git --version
```

- Python 3.12.3
- Git 2.43.0

## 4. Set Up SSH Key for GitHub

```bash
ssh-keygen -t ed25519 -C "raviashwin87@gmail.com"
cat ~/.ssh/id_ed25519.pub
```

Add the public key to GitHub: **Settings > SSH and GPG keys > New SSH key**

Test connection:

```bash
ssh -T git@github.com
```

## 5. Limit OpenBLAS Threads

The pip-installed NumPy/SciPy ship with OpenBLAS (`MAX_THREADS=64`). On a many-core VM this causes massive thread oversubscription for workloads with many small matrix operations (e.g. LRX per-pixel `np.linalg.solve`). Set single-threaded BLAS:

```bash
echo 'export OPENBLAS_NUM_THREADS=1' >> ~/.bashrc
source ~/.bashrc
```

## 6. Clone Repo

```bash
git clone git@github.com:rashwin88/hsi-anomaly-foundations-allotrope.git
```
