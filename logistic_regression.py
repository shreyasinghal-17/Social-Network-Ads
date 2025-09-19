# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from matplotlib.colors import ListedColormap

# # from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
# # from sklearn.pipeline import Pipeline
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import (
# #     confusion_matrix, classification_report,
# #     roc_auc_score, roc_curve, precision_recall_curve, auc
# # )
# # df = pd.read_csv('Social_Network_Ads.csv')
# # X = df.iloc[:, [2, 3]].values   
# # y = df.iloc[:, 4].values     

# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.25, stratify=y, random_state=0
# # )

# # pipe = Pipeline([
# #     ('scaler', StandardScaler()),
# #     ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=0))
# # ])

# # param_grid = {
# #     'clf__C': [0.01, 0.1, 1, 10, 100],
# #     'clf__penalty': ['l1', 'l2']  
# # }
# # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# # grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
# # grid.fit(X_train, y_train)

# # best_model = grid.best_estimator_
# # print("Best params:", grid.best_params_)

# # cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
# # print("CV ROC AUC: mean = {:.3f}, std = {:.3f}".format(cv_scores.mean(), cv_scores.std()))

# # y_pred = best_model.predict(X_test)
# # y_proba = best_model.predict_proba(X_test)[:, 1]

# # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# # print("\nClassification Report:\n", classification_report(y_test, y_pred))
# # print("Test ROC AUC:", roc_auc_score(y_test, y_proba))

# # fpr, tpr, _ = roc_curve(y_test, y_proba)
# # roc_auc = auc(fpr, tpr)
# # plt.figure()
# # plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
# # plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# # plt.legend()
# # plt.show()

# # prec, rec, _ = precision_recall_curve(y_test, y_proba)
# # pr_auc = auc(rec, prec)
# # plt.figure()
# # plt.plot(rec, prec, label=f'PR AUC = {pr_auc:.3f}')
# # plt.xlabel('Recall')
# # plt.ylabel('Precision')
# # plt.title('Precision-Recall Curve')
# # plt.legend()
# # plt.show()

# # x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
# # y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000
# # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
# #                      np.linspace(y_min, y_max, 300))
# # grid_points = np.c_[xx.ravel(), yy.ravel()]

# # Z = best_model.predict(grid_points)
# # Z = Z.reshape(xx.shape)

# # plt.figure(figsize=(9, 6))
# # plt.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(('red', 'green')))

# # plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='o', edgecolor='k', label='Train 0')
# # plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='s', edgecolor='k', label='Train 1')
# # plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], facecolors='none', edgecolor='k', marker='o', label='Test 0')
# # plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], facecolors='none', edgecolor='k', marker='s', label='Test 1')

# # plt.xlabel('Age')
# # plt.ylabel('Estimated Salary')
# # plt.title('Decision Boundary (original feature units)')
# # plt.legend()
# # plt.show()
# # clf = best_model.named_steps['clf']
# # print("Coefficients:", clf.coef_[0])
# # print("Intercept:", clf.intercept_[0])
# # print("Odds ratios (exp(coef)):", np.exp(clf.coef_[0]))
# # train_logistic_with_more_plots.py
# # Generates many diagnostic plots (heatmaps, pairplot, jointplot, box/violin/KDE,
# # KS plot, lift & cumulative gains, learning/validation curves, permutation importance).
# # Requires: numpy, pandas, matplotlib, seaborn, scikit-learn (>=0.24 recommended)
# # Run: python train_logistic_with_more_plots.py

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import seaborn as sns

# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve, validation_curve
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     confusion_matrix, classification_report,
#     roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score
# )
# from sklearn.calibration import calibration_curve
# from sklearn.inspection import permutation_importance

# # Create figures directory
# os.makedirs('figures', exist_ok=True)

# # ---------- Load data ----------
# df = pd.read_csv('Social_Network_Ads.csv')

# # If column names differ, adjust these lines:
# # We assume columns: [some id?, Age, EstimatedSalary, Purchased] or similar.
# # In your earlier code you used iloc[:, [2,3]] and iloc[:,4]:
# X = df.iloc[:, [2, 3]].copy().values   # Age, EstimatedSalary
# y = df.iloc[:, 4].copy().values        # Purchased

# # For plotting convenience convert X back to DataFrame with column names
# feat_names = ['Age', 'EstimatedSalary']
# X_df = pd.DataFrame(X, columns=feat_names)
# data = X_df.copy()
# data['Purchased'] = y

# # ---------- Train-test split ----------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, stratify=y, random_state=0
# )

# # ---------- Pipeline & GridSearch ----------
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=0))
# ])

# param_grid = {
#     'clf__C': [0.01, 0.1, 1, 10, 100],
#     'clf__penalty': ['l1', 'l2']
# }
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
# grid.fit(X_train, y_train)

# best_model = grid.best_estimator_
# print("Best params:", grid.best_params_)
# cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
# print("CV ROC AUC: mean = {:.3f}, std = {:.3f}".format(cv_scores.mean(), cv_scores.std()))

# # ---------- Predictions ----------
# y_pred = best_model.predict(X_test)
# y_proba = best_model.predict_proba(X_test)[:, 1]
# print("\nConfusion Matrix (threshold=0.5):\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("Test ROC AUC:", roc_auc_score(y_test, y_proba))

# # ---------- Basic statistical and EDA plots ----------
# sns.set(style="whitegrid")

# # Correlation heatmap (works well when more numerical features are added)
# plt.figure(figsize=(6,5))
# corr = data[feat_names].corr()
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title('Feature Correlation Heatmap')
# plt.savefig('figures/corr_heatmap.png', bbox_inches='tight')
# plt.close()

# # Pairplot (may be slow for large datasets)
# pairplot = sns.pairplot(data, vars=feat_names, hue='Purchased', height=3, diag_kind='hist', plot_kws={'alpha':0.6})
# pairplot.fig.suptitle('Pairplot: Features colored by Purchased', y=1.02)
# plt.savefig('figures/pairplot.png', bbox_inches='tight')
# plt.close()

# # Jointplot Age vs Salary
# jp = sns.jointplot(x='Age', y='EstimatedSalary', data=data, hue='Purchased', kind='scatter', height=6, marginal_kws=dict(bins=20, fill=True))
# jp.fig.suptitle('Jointplot: Age vs Estimated Salary', y=1.02)
# plt.savefig('figures/joint_age_salary.png', bbox_inches='tight')
# plt.close()

# # Boxplots by class
# fig, axs = plt.subplots(1, len(feat_names), figsize=(10,4))
# for i, f in enumerate(feat_names):
#     sns.boxplot(x='Purchased', y=f, data=data, ax=axs[i])
#     axs[i].set_title(f'Boxplot of {f} by Purchased')
# plt.tight_layout()
# plt.savefig('figures/boxplots.png', bbox_inches='tight')
# plt.close()

# # Violin plots by class
# fig, axs = plt.subplots(1, len(feat_names), figsize=(10,4))
# for i, f in enumerate(feat_names):
#     sns.violinplot(x='Purchased', y=f, data=data, ax=axs[i], inner='quartile')
#     axs[i].set_title(f'Violin plot of {f} by Purchased')
# plt.tight_layout()
# plt.savefig('figures/violinplots.png', bbox_inches='tight')
# plt.close()

# # KDE plots by class for each feature
# plt.figure(figsize=(10,4))
# for i, f in enumerate(feat_names):
#     plt.subplot(1, len(feat_names), i+1)
#     sns.kdeplot(data=data, x=f, hue='Purchased', fill=True, common_norm=False, alpha=0.5)
#     plt.title(f'KDE of {f}')
# plt.tight_layout()
# plt.savefig('figures/kde_by_class.png', bbox_inches='tight')
# plt.close()

# # ---------- ROC, PR (again, saved earlier but keep here) ----------
# fpr, tpr, _ = roc_curve(y_test, y_proba)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(6,6))
# plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
# plt.plot([0,1],[0,1],'k--', linewidth=0.8)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.savefig('figures/roc_curve.png', bbox_inches='tight')
# plt.close()

# prec, rec, _ = precision_recall_curve(y_test, y_proba)
# pr_auc = auc(rec, prec)
# plt.figure(figsize=(6,6))
# plt.plot(rec, prec, label=f'PR AUC = {pr_auc:.3f}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower left')
# plt.savefig('figures/pr_curve.png', bbox_inches='tight')
# plt.close()

# # ---------- KS plot (TPR - FPR vs threshold) ----------
# # compute TPR/FPR across thresholds from roc_curve
# fpr_all, tpr_all, thresholds_all = roc_curve(y_test, y_proba)
# ks_stat = tpr_all - fpr_all
# best_ks_idx = np.argmax(ks_stat)
# best_ks = ks_stat[best_ks_idx]
# best_thresh_ks = thresholds_all[best_ks_idx]
# plt.figure(figsize=(7,5))
# plt.plot(thresholds_all, tpr_all, label='TPR')
# plt.plot(thresholds_all, fpr_all, label='FPR')
# plt.plot(thresholds_all, ks_stat, label='TPR - FPR (KS)', linestyle='--')
# plt.axvline(best_thresh_ks, color='k', linestyle=':', label=f'best KS thresh={best_thresh_ks:.2f}')
# plt.xlabel('Threshold')
# plt.ylabel('Rate')
# plt.title(f'KS Plot (best KS={best_ks:.3f} at thresh={best_thresh_ks:.2f})')
# plt.legend()
# plt.savefig('figures/ks_plot.png', bbox_inches='tight')
# plt.close()

# # ---------- Lift chart & Cumulative gains ----------
# # Build DataFrame of y_true and y_proba
# df_scores = pd.DataFrame({'y_true': y_test, 'y_proba': y_proba})
# df_scores = df_scores.sort_values('y_proba', ascending=False).reset_index(drop=True)
# df_scores['cum_positives'] = df_scores['y_true'].cumsum()
# df_scores['n'] = np.arange(1, len(df_scores) + 1)
# total_positives = df_scores['y_true'].sum()
# df_scores['cum_gain'] = df_scores['cum_positives'] / total_positives
# # deciles
# df_scores['decile'] = pd.qcut(df_scores['n'], 10, labels=False)
# lift_df = df_scores.groupby('decile').agg({'y_true': ['sum', 'count']})
# lift_df.columns = ['positives', 'count']
# lift_df = lift_df.reset_index().sort_values('decile', ascending=True)
# lift_df['rate'] = lift_df['positives'] / lift_df['count']
# overall_rate = total_positives / len(df_scores)
# lift_df['lift'] = lift_df['rate'] / overall_rate

# # Plot lift chart
# plt.figure(figsize=(8,5))
# plt.plot(range(1,11), lift_df['lift'], marker='o')
# plt.xlabel('Decile (1 = top scores)')
# plt.ylabel('Lift')
# plt.title('Lift Chart by Decile')
# plt.grid(True)
# plt.savefig('figures/lift_chart.png', bbox_inches='tight')
# plt.close()

# # Plot cumulative gains
# plt.figure(figsize=(7,5))
# plt.plot(df_scores['n'], df_scores['cum_gain'], label='Model')
# plt.plot([0, len(df_scores)], [0, 1], 'k--', label='Random')
# plt.xlabel('Number of samples (sorted by probability)')
# plt.ylabel('Cumulative fraction of positives captured')
# plt.title('Cumulative Gains Curve')
# plt.legend()
# plt.savefig('figures/cumulative_gain.png', bbox_inches='tight')
# plt.close()

# # ---------- Learning curve (ROC AUC) ----------
# train_sizes, train_scores, test_scores = learning_curve(
#     best_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
#     train_sizes=np.linspace(0.1, 1.0, 5), random_state=0
# )
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.figure(figsize=(8,6))
# plt.plot(train_sizes, train_mean, marker='o', label='Training ROC AUC')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
# plt.plot(train_sizes, test_mean, marker='o', label='Validation ROC AUC')
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
# plt.xlabel('Training set size')
# plt.ylabel('ROC AUC')
# plt.title('Learning Curve (ROC AUC)')
# plt.legend()
# plt.grid(True)
# plt.savefig('figures/learning_curve.png', bbox_inches='tight')
# plt.close()

# # ---------- Validation curve for parameter C ----------
# param_range = np.logspace(-2, 2, 5)
# train_scores_vc, test_scores_vc = validation_curve(
#     pipe, X, y, param_name='clf__C', param_range=param_range, cv=cv, scoring='roc_auc', n_jobs=-1
# )
# train_mean_vc = np.mean(train_scores_vc, axis=1)
# test_mean_vc = np.mean(test_scores_vc, axis=1)

# plt.figure(figsize=(7,5))
# plt.semilogx(param_range, train_mean_vc, marker='o', label='Training ROC AUC')
# plt.semilogx(param_range, test_mean_vc, marker='o', label='Validation ROC AUC')
# plt.xlabel('C (inverse regularization strength)')
# plt.ylabel('ROC AUC')
# plt.title('Validation Curve for C')
# plt.legend()
# plt.grid(True)
# plt.savefig('figures/validation_curve_C.png', bbox_inches='tight')
# plt.close()

# # ---------- Permutation importance ----------
# try:
#     r = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=0, n_jobs=-1, scoring='roc_auc')
#     imp_means = r.importances_mean
#     imp_std = r.importances_std
#     idx = np.argsort(imp_means)[::-1]
#     plt.figure(figsize=(6,4))
#     sns.barplot(x=imp_means[idx], y=np.array(feat_names)[idx], xerr=imp_std[idx])
#     plt.xlabel('Mean decrease in ROC AUC (permutation importance)')
#     plt.title('Permutation Feature Importance')
#     plt.savefig('figures/permutation_importance.png', bbox_inches='tight')
#     plt.close()
# except Exception as e:
#     print("Permutation importance failed:", e)

# # ---------- Save classification report as text ----------
# report = classification_report(y_test, y_pred)
# with open('figures/classification_report.txt', 'w') as f:
#     f.write(report)

# print("Generated many diagnostic figures in the 'figures/' directory.")



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
