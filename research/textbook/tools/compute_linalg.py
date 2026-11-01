"""Book chapter ch:linalg - every worked example, checked with numpy.
Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_linalg.py > research/textbook/tools/compute_linalg_output.txt"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

print("== dot, norm, angle (IW08 numbers)")
a = np.array([0.1, 0.2, 0.3]); b = np.array([0.3, 0.2, 0.1])
print("a.b", a @ b, "|a|", np.linalg.norm(a), "|b|", np.linalg.norm(b))
c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("cos", c, "theta rad", np.arccos(c), "deg", np.degrees(np.arccos(c)))
print("2a angle", np.degrees(np.arccos(np.clip(a @ (2 * a) / (np.linalg.norm(a) * np.linalg.norm(2 * a)), -1, 1))))

print("== projection 2-D")
x = np.array([3.0, 1.0]); u = np.array([1.0, 1.0])
t = x @ u / (u @ u); p = t * u; r = x - p
print("t", t, "proj", p, "resid", r, "r.u", r @ u, "|r|^2", r @ r)
for tt in [1.0, 1.5, 2.0, 2.5, 3.0]:
    print("  t", tt, "||x - t u||^2", np.sum((x - tt * u) ** 2))

print("== projection 3-band")
x = np.array([0.12, 0.24, 0.30]); u = np.array([0.1, 0.2, 0.3])
t = x @ u / (u @ u); p = t * u; r = x - p
print("x.u", x @ u, "u.u", u @ u, "t", t, "proj", p, "resid", r, "r.u", r @ u)
print("|x|^2", x @ x, "|p|^2", p @ p, "|r|^2", r @ r, "sum", p @ p + r @ r)

print("== matrix products")
A = np.array([[1, 2], [3, 4]]); B = np.array([[0, 1], [5, 2]])
print("AB", A @ B, "(AB)T", (A @ B).T, "BT AT", B.T @ A.T, "AT BT", A.T @ B.T, sep="\n")
print("A v", A @ np.array([1, -1]))

print("== 2x2 inverse")
M = np.array([[2.0, 3.0], [3.0, 10.0]])
print("det", np.linalg.det(M), "inv*11", np.linalg.inv(M) * 11, "check", M @ np.linalg.inv(M))

print("== 4 pixels x 2 bands")
X = np.array([[8, 18], [10, 17], [11, 24], [11, 21]], float)
mu = X.mean(0); D = X - mu
print("mean", mu); print("dev", D)
S = sum(np.outer(d, d) for d in D)
print("sum outer", S, "cov (N-1)", S / 3, "np.cov", np.cov(X.T), "pop (N)", S / 4, sep="\n")
C = np.cov(X.T)
ev, U = np.linalg.eigh(C)
print("eig", ev, "U", U, sep="\n")
U = np.array([[1, 3], [3, -1]]) / np.sqrt(10)   # columns u1=(1,3), u2=(3,-1) / sqrt 10
L = np.diag([11.0, 1.0])
print("U L UT", U @ L @ U.T)
print("UT U", U.T @ U)
print("explained", 11 / 12, 1 / 12)
proj = D @ U
print("coords on u1,u2 (x sqrt10)", proj * np.sqrt(10))
print("var of coords", proj.var(0, ddof=1), "cov", np.cov(proj.T))
W = np.diag(1 / np.sqrt([11.0, 1.0])) @ U.T
Z = D @ W.T
print("W", W); print("whitened", Z); print("cov whitened", np.cov(Z.T))
print("W S WT", W @ C @ W.T)
print("1/sqrt11", 1 / np.sqrt(11), "1/sqrt10", 1 / np.sqrt(10), "1/sqrt110", 1 / np.sqrt(110))
# variance along a few unit directions (for the PCA picture)
for ang in [0, 30, 45, 60, 71.565, 90, 135]:
    v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    print("  angle", ang, "u^T S u", v @ C @ v)

print("== illustrative cloud for the figures: 60 draws from N(0, [[2,3],[3,10]]), seed 0")
rng = np.random.default_rng(0)
P = rng.multivariate_normal([0, 0], C, size=60)
print("sample cov of the cloud", np.cov(P.T))
print("cloud", " ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in P))
Q = P @ W.T
print("whitened cloud", " ".join(f"({q[0]:.2f},{q[1]:.2f})" for q in Q))
