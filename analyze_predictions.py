import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import os

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', None)



if __name__ == '__main__':

    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/output/MakeModelClassifier/2021-10-26-14h28/logs'
    df = pd.read_csv(os.path.join(path, 'predictions.csv'))
    for col in df.columns[:-1]:
        df[col] = df[col].astype(float)

    classes = df[['true_label']].copy()
    del df['true_label']

    # Get top 5 predicted classes per observation
    lst = []
    index = df.columns.tolist()  # columns become indices below
    for i in range(len(df)):
        top_five = np.argsort(df.iloc[i].values)[-5:]
        names = list(reversed([index[i] for i in top_five]))
        lst.append(names)
    pred_classes = pd.DataFrame(lst, columns=['Argmax(0)', 'Argmax(1)', 'Argmax(2)', 'Argmax(3)', 'Argmax(4)'])

    # Concat true and predicted labels together
    classes = pd.concat([classes, pred_classes], axis=1)

    # Accuracy via argmax values
    one = np.where(classes['true_label'] == classes['Argmax(0)'], 1, 0)
    two = np.where((classes['true_label'] == classes['Argmax(0)']) |
                   (classes['true_label'] == classes['Argmax(1)']), 1, 0)
    three = np.where((classes['true_label'] == classes['Argmax(0)']) |
                     (classes['true_label'] == classes['Argmax(1)']) |
                     (classes['true_label'] == classes['Argmax(2)']), 1, 0)
    four = np.where((classes['true_label'] == classes['Argmax(0)']) |
                     (classes['true_label'] == classes['Argmax(1)']) |
                     (classes['true_label'] == classes['Argmax(2)']) |
                     (classes['true_label'] == classes['Argmax(3)']), 1, 0)
    five = np.where((classes['true_label'] == classes['Argmax(0)']) |
                     (classes['true_label'] == classes['Argmax(1)']) |
                     (classes['true_label'] == classes['Argmax(2)']) |
                     (classes['true_label'] == classes['Argmax(3)']) |
                     (classes['true_label'] == classes['Argmax(4)']), 1, 0)

    accuracy = pd.DataFrame()
    accuracy = pd.concat([accuracy, pd.DataFrame(one.mean(), columns=['Accuracy'], index=['Argmax(0)'])], axis=0)
    accuracy = pd.concat([accuracy, pd.DataFrame(two.mean(), columns=['Accuracy'], index=['Argmax(0:1)'])], axis=0)
    accuracy = pd.concat([accuracy, pd.DataFrame(three.mean(), columns=['Accuracy'], index=['Argmax(0:2)'])], axis=0)
    accuracy = pd.concat([accuracy, pd.DataFrame(four.mean(), columns=['Accuracy'], index=['Argmax(0:3)'])], axis=0)
    accuracy = pd.concat([accuracy, pd.DataFrame(five.mean(), columns=['Accuracy'], index=['Argmax(0:4)'])], axis=0)
    accuracy = accuracy.reset_index()

    plt.close()
    figure(figsize=(10, 8))
    g = sns.barplot(data=accuracy, x='index', y='Accuracy', palette='Set2')
    for index, row in accuracy.iterrows():  # print accuracy values atop each bar
        g.text(row.name, row.Accuracy, round(row.Accuracy, 4), color='black', ha='center')
    plt.xlabel(None)
    plt.ylabel('Categorical Accuracy')
    plt.title('Accuracy Among Top 5 Predicted Classes')
    plt.savefig(os.path.join(path, 'Accuracy_Top5.png'))
    plt.close()

    # Multiclass confusion matrix
    labels = classes['true_label'].drop_duplicates().sort_values().tolist()
    conf_mat = pd.DataFrame(confusion_matrix(classes['true_label'], classes['Argmax(0)'], normalize='true', labels=labels), index=labels, columns=labels)
    conf_mat.to_csv(os.path.join(path, 'confusion_matrix.csv'))

    # Output heatmap of confusion matrix
    figure(figsize=(25, 25))
    sns.set(font_scale=0.5)
    ax = sns.heatmap(conf_mat, cmap='Reds', linewidth=0.8, cbar_kws={"shrink": 0.8}, square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title('Confusion Matrix Heatmap', fontsize=30)
    plt.savefig(os.path.join(path, 'heatmap.png'))
    plt.close()


    # One vs rest confusion matrix
    ml_conf_mat = multilabel_confusion_matrix(classes['true_label'], classes['Argmax(0)'], labels=labels, samplewise=False)  # class-wise confusion matrix
    lst = []
    for i in range(len(ml_conf_mat)):
        temp = ml_conf_mat[i]
        tnr, fpr, fnr, tpr = temp.ravel()
        lst.append([tnr, fpr, fnr, tpr])
    ovr_conf_mat = pd.DataFrame(lst, columns=['TN', 'FP', 'FN', 'TP'], index=labels)
    ovr_conf_mat = ovr_conf_mat[['TP', 'FN', 'FP', 'TN']]
    ovr_conf_mat['Accuracy'] = (ovr_conf_mat['TP'] +  ovr_conf_mat['TN']) / ovr_conf_mat.sum(axis=1).mean()
    ovr_conf_mat['Precision'] = ovr_conf_mat['TP'] / (ovr_conf_mat['TP'] + ovr_conf_mat['FP'])
    ovr_conf_mat['Recall_Sensitivity_TPR'] = ovr_conf_mat['TP'] / (ovr_conf_mat['TP'] + ovr_conf_mat['FN'])
    ovr_conf_mat['FNR'] = ovr_conf_mat['FN'] / (ovr_conf_mat['FN'] + ovr_conf_mat['TP'])
    ovr_conf_mat['FPR'] = ovr_conf_mat['FP'] / (ovr_conf_mat['FP'] + ovr_conf_mat['TN'])
    ovr_conf_mat['Specificity_TNR'] = ovr_conf_mat['TN'] / (ovr_conf_mat['TN'] + ovr_conf_mat['FP'])
    ovr_conf_mat['F1'] = 2 * ovr_conf_mat['TP'] / ((2 * ovr_conf_mat['TP']) + ovr_conf_mat['FP'] + ovr_conf_mat['FN'])
    for col in ovr_conf_mat.columns[4:]:
        ovr_conf_mat[col] = round(ovr_conf_mat[col], 4)
    ovr_conf_mat.to_csv(os.path.join(path, 'OVR Confusion Matrix.csv'))

    # Kernel density plot of sensitivity
    figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.kdeplot(ovr_conf_mat['Recall_Sensitivity_TPR'])
    plt.xlabel('Sensitivity / Recall / TPR')
    plt.title("Kernel Density of Sensitivity")
    plt.savefig(os.path.join(path, 'sensitivity_kdeplot.png'))
    plt.close()

    sens = ovr_conf_mat[['Recall_Sensitivity_TPR']].copy()
    sens = sens.sort_values(by=['Recall_Sensitivity_TPR'], ascending=False).reset_index()
    combined = pd.concat([sens.iloc[:50, :], sens.iloc[-50:, :]], axis=0).reset_index(drop=True)

    # Figure to output barplot of sensitivity among best and worst 50 classified make-models
    plt.close()
    figure(figsize=(20, 8))
    sns.set(font_scale=0.8)
    ax = sns.barplot(data=combined, y='Recall_Sensitivity_TPR', x='index', saturation=0.9)
    plt.xticks(rotation=70, ha='right')
    plt.yticks(fontsize=10)
    plt.xlabel(None)
    plt.ylabel('Sensitivity / Recall / TPR', fontsize=15)
    plt.tight_layout()
    plt.title('Best and Worst 50 Classified Make-Models', fontsize=20, pad=-18)
    plt.savefig(os.path.join(path, 'sensitivity_bar.png'), dpi=200)
    plt.close()
    
