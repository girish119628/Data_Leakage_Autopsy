"""Reference code for every notebook cell. Run on churn_data.csv and compare to the script targets."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import make_data

df = pd.read_csv("churn_data.csv")
TARGET, PROXY = "churned_next_8m", "cancellation_ticket_opened"
print("shape", df.shape, "| churn rate", round(df[TARGET].mean(), 3),
      "| majority-class accuracy", round(1 - df[TARGET].mean(), 3), "| customers", df.customer_id.nunique())
print("missing:", df.isna().sum()[df.isna().sum() > 0].to_dict())

d = pd.get_dummies(df, columns=["plan_type", "payment_method", "region"], dtype=int)
y = d[TARGET]; groups = d["customer_id"]; month = d["snapshot_month"]
X_all = d.drop(columns=[TARGET, "customer_id", "snapshot_month"])
rf = lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("rf", rf())])
kf = KFold(5, shuffle=True, random_state=42)

# Cell 1: stacked leaky model
Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_all))
print("\nCell 1 stacked leaky      ", round(cross_val_score(rf(), Xs, y, cv=kf).mean() * 100, 1))
# Cell 2: pipeline fix (proxy still in)
print("Cell 2 pipeline           ", round(cross_val_score(pipe(), X_all, y, cv=kf).mean() * 100, 1))

# Cell 3: single-feature AUC scan, then drop proxy
auc = {c: cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), X_all[[c]], y,
                          cv=kf, scoring="roc_auc").mean() for c in X_all.columns}
print("Cell 3 single-feature AUC top 5:", {k: round(v, 3) for k, v in sorted(auc.items(), key=lambda t: -t[1])[:5]})
X = X_all.drop(columns=[PROXY])
print("Cell 3 drop proxy (KFold) ", round(cross_val_score(pipe(), X, y, cv=kf).mean() * 100, 1))

# Cell 4: GroupKFold
print("Cell 4 GroupKFold         ", round(cross_val_score(pipe(), X, y, cv=GroupKFold(5), groups=groups).mean() * 100, 1))

# Cell 5: time split
tr, te = month <= 24, month >= 25
p = pipe().fit(X[tr], y[tr])
print("Cell 5 time split         ", round(p.score(X[te], y[te]) * 100, 1), "| n_train", tr.sum(), "n_test", te.sum())

# Cell 6: noise demo
Xn, yn = make_data.make_noise()
kfn = KFold(5, shuffle=True, random_state=0)
lr = LogisticRegression(max_iter=1000)
leaky = cross_val_score(lr, SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=kfn).mean()
honest = cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=20)), ("lr", lr)]), Xn, yn, cv=kfn).mean()
print("Cell 6 noise: leaky", round(leaky * 100, 1), "| inside Pipeline", round(honest * 100, 1), "| shape", Xn.shape)

# Cell 7: adversarial validation
def adv_auc(Xa, Xb):
    Z = pd.concat([Xa, Xb]); lab = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    Zi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Z), columns=Z.columns)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    s = cross_val_score(clf, Zi, lab, cv=5, scoring="roc_auc").mean()
    imp = pd.Series(clf.fit(Zi, lab).feature_importances_, index=Z.columns).sort_values(ascending=False)
    return s, imp
a_rand, imp_r = adv_auc(*[X.iloc[i] for i in train_test_split(np.arange(len(X)), test_size=0.3, random_state=1)])
a_time, imp_t = adv_auc(X[tr], X[te])
print("Cell 7 adversarial AUC random split", round(a_rand, 3), "| time split", round(a_time, 3),
      "| top features (time):", list(imp_t.index[:3]))

# Cell 8: label shuffle (procedure test) on the noise pipeline, labels reshuffled
rng = np.random.default_rng(0); ysh = rng.permutation(yn)
l2 = cross_val_score(lr, SelectKBest(f_classif, k=20).fit_transform(Xn, ysh), ysh, cv=kfn).mean()
h2 = cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=20)), ("lr", lr)]), Xn, ysh, cv=kfn).mean()
print("Cell 8 shuffled labels: leaky procedure", round(l2 * 100, 1), "| honest pipeline", round(h2 * 100, 1))
# also: shuffle test on the churn pipeline (honest) should collapse to chance
ys = pd.Series(rng.permutation(y.values), index=y.index)
print("Cell 8 churn pipeline, shuffled labels, AUC:",
      round(cross_val_score(pipe(), X, ys, cv=GroupKFold(5), groups=groups, scoring="roc_auc").mean(), 3))
