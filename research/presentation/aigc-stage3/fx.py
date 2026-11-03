p = "journey.tex"
s = open(p, encoding="utf8").read()
def rep(a, b):
    global s
    assert s.count(a) >= 1, a[:60]
    s = s.replace(a, b)
for a, b in [("(-1,-35) rectangle (34,27.5)", "(-1,-31) rectangle (34,25)"), ("(34.6,-35) rectangle (60,27.5)", "(34.6,-31) rectangle (60,25)"),
             ("(60.6,-35) rectangle (121,27.5)", "(60.6,-31) rectangle (121,25)"), ("(121.6,-35) rectangle (149,27.5)", "(121.6,-31) rectangle (149,25)")]:
    rep(a, b)
rep(r"at (\x,25.3) {\t};", r"at (\x,22.8) {\t};")
rep(r"{10/4/17mm/{\textbf{23 Mar} first\\trainer, S3 shards}}", r"{10/4/9mm/{\textbf{23 Mar}\\first\\trainer}}")
rep(r"{68/3/11mm/{\textbf{15 Oct}\\residual\\Mahalanobis}}", r"{66/3/13mm/{\textbf{15 Oct}\\residual\\Mahalanobis}}")
rep(r"{124/-19/13mm/", r"{124/-17/13mm/")
rep(r"(146,-3.3) -- (146,-19);", r"(146,-3.3) -- (146,-17);")
rep(r"at (147.5,-19.6) {\textbf{1 Nov}", r"at (147.5,-17.6) {\textbf{1 Nov}")
rep(r"\begin{scope}[shift={(66,-29.5)}]", r"\begin{scope}[shift={(66,-26)}]")
open(p, "w", encoding="utf8").write(s)
m = open("main.tex", encoding="utf8").read()
a = "\\input{journey}"
i = m.index(a)
j = m.index("\\vspace{0.8mm}", i)
m = m[:j] + "\\vspace{0.3mm}" + m[j + len("\\vspace{0.8mm}"):]
open("main.tex", "w", encoding="utf8").write(m)
print("ok")
