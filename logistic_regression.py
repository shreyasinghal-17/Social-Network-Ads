import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance

os.makedirs('figures', exist_ok=True)

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values
feat_names = ['Age', 'EstimatedSalary']
X_df = pd.DataFrame(X, columns=feat_names)
data = X_df.copy()
data['Purchased'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=0))])
param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('figures/roc_curve.png', bbox_inches='tight', dpi=200)
plt.close()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(rec, prec)
plt.figure(figsize=(6,6))
plt.plot(rec, prec, label=f'PR AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('figures/pr_curve.png', bbox_inches='tight', dpi=200)
plt.close()

x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = best_model.predict(grid_points).reshape(xx.shape)
plt.figure(figsize=(9,6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(('red','green')))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='o', edgecolor='k', label='Train 0', alpha=0.9)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='s', edgecolor='k', label='Train 1', alpha=0.9)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], facecolors='none', edgecolor='k', marker='o', label='Test 0')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], facecolors='none', edgecolor='k', marker='s', label='Test 1')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Decision Boundary (class labels)')
plt.legend()
plt.savefig('figures/decision_boundary.png', bbox_inches='tight', dpi=200)
plt.close()

Z_proba = best_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
plt.figure(figsize=(9,6))
cont = plt.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlGn', alpha=0.8)
plt.colorbar(cont, label='P(y=1)')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='o', edgecolor='k', label='Train 0', alpha=0.9)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='s', edgecolor='k', label='Train 1', alpha=0.9)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Decision Boundary (predicted probability P(y=1))')
plt.legend()
plt.savefig('figures/decision_boundary_proba.png', bbox_inches='tight', dpi=200)
plt.close()

pairplot = sns.pairplot(data, vars=feat_names, hue='Purchased', height=3, diag_kind='hist', plot_kws={'alpha':0.6})
pairplot.fig.suptitle('Pairplot: Features colored by Purchased', y=1.02)
pairplot.fig.savefig('figures/pairplot.png', bbox_inches='tight', dpi=200)
plt.close()

plt.figure(figsize=(6,5))
corr = data[feat_names].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Feature Correlation Heatmap')
plt.savefig('figures/corr_heatmap.png', bbox_inches='tight', dpi=200)
plt.close()

def save_confusion_matrix(y_true, y_proba, threshold, filename):
    y_pred_t = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_t)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (threshold={threshold:.2f})')
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()

save_confusion_matrix(y_test, y_proba, 0.30, 'figures/confusion_matrix_t30.png')
save_confusion_matrix(y_test, y_proba, 0.50, 'figures/confusion_matrix_t50.png')
save_confusion_matrix(y_test, y_proba, 0.70, 'figures/confusion_matrix_t70.png')

with open('figures/classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))

