import numpy as np
np.random.seed(42)
from numpy import inf
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
import os
#import shap
os.environ["KMP_WARNINGS"] = "FALSE"
######## Import some multi-output models #######################################
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain, ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics

import copy
from inspect import signature
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

import matplotlib.pylab as plt
# get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = (10, 6)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle

##############################################################################################

def multi_f1(truth, predictions):
    return f1_score(truth, predictions, average=None)


def multi_precision(truth, predictions):
    return precision_score(truth, predictions, average=None)


##############################################################################################
def Draw_MC_ML_PR_ROC_Curves(classifier, X_test, y_test):
    """
    ========================================================================================
    Precision-Recall Curves: Extension of Original Version in SKLearn's Documentation Page:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    ========================================================================================
    """
    figsize = (10, 6)
    ###############################################################################
    # In binary classification settings
    # Compute the average precision score for Binary Classes
    ###############################################################################
    y_pred = classifier.predict(X_test)
    if len(left_subtract(np.unique(y_test), np.unique(y_pred))) == 0:
        classes = list(range(len(np.unique(y_test))))
    else:
        classes = list(range(len(np.unique(y_pred))))
    # classes = list(range(len(np.unique(y_test))))
    n_classes = len(classes)
    if n_classes == 2:
        try:
            y_score = classifier.decision_function(X_test)
        except:
            y_score = classifier.predict_proba(X_test)
        try:
            average_precision = average_precision_score(y_test, y_score)
        except:
            average_precision = multi_precision(y_test, classifier.predict(X_test)).mean()
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
        f_scores = multi_f1(y_test, classifier.predict(X_test))
        print('Macro F1 score, averaged over all classes: {0:0.2f}'
              .format(f_scores.mean()))
        ###############################################################################
        # Plot the Precision-Recall curve
        # ................................
        plt.figure(figsize=figsize)
        try:
            ### This works for Logistic Regression and other Linear Models ####
            precision, recall, _ = precision_recall_curve(y_test, y_score)
        except:
            ### This works for Non Linear Models such as Forests and XGBoost #####
            precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='g', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='g', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Avg.Precision={0:0.2f}, Avg F1={1:0.2f}'.format(
            average_precision, f_scores.mean()))
        plt.show()
        ###############################################################################
    else:
        # In multi-label settings
        # ------------------------
        #
        # Create multi-label data, fit, and predict
        # ...........................................
        #
        # We create a multi-label dataset, to illustrate the precision-recall in
        # multi-label settings

        # Use label_binarize to be multi-label like settings
        Y = label_binarize(y_test, classes)
        n_classes = Y.shape[1]
        Y_test = copy.deepcopy(Y)
        try:
            y_score = classifier.decision_function(X_test)
        except:
            y_score = classifier.predict_proba(X_test)

        ###############################################################################
        # The average precision score in multi-label settings
        # ....................................................

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test[
                                                                        :, :n_classes].ravel(),
                                                                        y_score[:, :n_classes].ravel())
        average_precision["micro"] = average_precision_score(Y_test[
                                                             :, :n_classes], y_score[:, :n_classes], average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))

        ###############################################################################
        # Plot Precision-Recall curve for each class and iso-f1 curves
        # Plot the micro-averaged Precision-Recall curve
        ###############################################################################
        # .............................................................
        #
        # setup plot details
        colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')

        plt.figure(figsize=figsize)
        f_scores = multi_f1(y_test, classifier.predict(X_test))
        print('Macro F1 score, averaged over all classes: {0:0.2f}'
              .format(f_scores.mean()))
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=1)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Micro Avg Precision-Recall curve with iso-f1 curves')
        plt.legend(lines, labels, loc='lower left', prop=dict(size=10))
        plt.show()

#####################################################################
#####     REGRESSION CHARTS AND METRICS ARE PRINTED PLOTTED HERE
#####################################################################
def plot_regression_scatters(df, df2, num_vars, plot_name=''):
    """
    Great way to plot continuous variables fast. Just sent them in and it will take care of the rest!
    """
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    col = 2
    start_time = time.time()
    row = len(num_vars)
    fig, ax = plt.subplots(row, col)
    if col < 2:
        fig.set_size_inches(min(15, 8), row * 5)
        fig.subplots_adjust(hspace=0.5)  ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3)  ### This controls the space between columns
    else:
        fig.set_size_inches(min(col * 10, 20), row * 5)
        fig.subplots_adjust(hspace=0.3)  ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3)  ### This controls the space between columns
    fig.suptitle('Regression Metrics Plots for %s Model' % plot_name, fontsize=20)
    counter = 0
    if row == 1:
        ax = ax.reshape(-1, 1).T
    for k in np.arange(row):
        row_color = next(colors)
        for l in np.arange(col):
            try:
                if col == 1:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:, k]
                        y = df2[:, k]
                    ax1 = ax[k][l]
                    lineStart = x.min()
                    lineEnd = x.max()
                    ax1.scatter(x, y, color=row_color)
                    ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=row_color)
                    ax1.set_xlabel('Actuals')
                    ax1.set_ylabel('Predicted')
                    ax1.set_title('Predicted vs Actuals Plot for Target = %s' % num_vars[k])
                else:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:, k]
                        y = df2[:, k]
                    lineStart = x.min()
                    lineEnd = x.max()
                    if l == 0:
                        ax1 = ax[k][l]
                        ax1.scatter(x, y, color=row_color)
                        ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=row_color)
                        ax1.set_xlabel('Actuals')
                        ax1.set_ylabel('Predicted')
                        ax1.set_title('Predicted vs Actuals Plot for Target = %s' % num_vars[k])
                    else:
                        ax1 = ax[k][l]
                        ax1.hist((x - y), density=True, color=row_color)
                        ax1.axvline(color='k')
                        ax1.set_title('Residuals Plot for Target = %s' % num_vars[k])
            except:
                if col == 1:
                    counter += 1
                else:
                    ax[k][l].set_title('No Predicted vs Actuals Plot for plot as %s is not numeric' % num_vars[k])
                    counter += 1
    print('Regression Plots completed in %0.3f seconds' % (time.time() - start_time))

######################################################################################
def draw_confusion_maxtrix(y_test, y_pred, model_name='Model', ax=''):
    """
    This plots a beautiful confusion matrix based on input: ground truths and predictions
    """
    # Confusion Matrix
    '''Plotting CONFUSION MATRIX'''
    import seaborn as sns
    sns.set_style('darkgrid')

    '''Display'''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    pd.options.display.float_format = '{:,.2f}'.format

    # Get the confusion matrix and put it into a df
    from sklearn.metrics import confusion_matrix, f1_score

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,
                         index=np.unique(y_test).tolist(),
                         columns=np.unique(y_test).tolist(),
                         )

    sns.heatmap(cm_df,
                center=0,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),
                annot=True,
                fmt='g',
                ax=ax)

    ax.set_title(' %s \nF1 Score(avg = micro): %0.2f \nF1 Score(avg = macro): %0.2f' % (
        model_name, f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro')),
                 fontsize=13)
    ax.set_ylabel('True label', fontsize=13)
    ax.set_xlabel('Predicted label', fontsize=13)

######################################################################################
def plot_precision_recall_curve(m, x_testc, y_test, ax=None):
    #### This is a simple way to plot PR curve for binary classes #####
    y_scores = m.predict_proba(x_testc)[:, 1]
    ### generate the precision recall curve ##########
    p, r, thresholds = precision_recall_curve(y_test, y_scores)
    #### plot the curve ##########
    ax.set_title("Precision Recall Curve")
    ax.plot(r, p, color="purple")
    ax.set_ylim([0.5, 1.01])
    ax.set_xlim([0.5, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')


def plot_roc_curve(m, x_testc, y_test, ax=None):
    #### This is a simple way to plot ROC curve for binary classes #####
    y_scores = m.predict_proba(x_testc)[:, 1]
    ### generate the precision recall curve ##########
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    #### plot the curve ##########
    ax.set_title("ROC AUC Curve")
    ax.plot(fpr, tpr, color="purple")
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

######################################################################################
def plot_classification_results(m, X_true, y_true, each_target):
    #### These plots are only for binary classes #############
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        plot_roc_curve(m, X_true, y_true, ax=axes[0, 1])
        axes[0, 1].set_title('ROC AUC Curve: %s' % each_target)

        plot_precision_recall_curve(m, X_true, y_true, ax=axes[1, 0])
        axes[1, 0].set_title('PR AUC Curve for: %s' % each_target)
        y_pred = m.predict(X_true)
        draw_confusion_maxtrix(y_true, y_pred, 'Confusion Matrix', ax=axes[0, 0])
        try:
            clf_report = classification_report(y_true,
                                               y_pred,
                                               # labels=target_names,
                                               # target_names=labels,
                                               output_dict=True)
        except:
            clf_report = classification_report(y_true, y_pred,
                                               # labels=target_names,
                                               # target_names=labels,
                                               output_dict=True)

        sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True, ax=axes[1, 1], fmt='0.2f')
        axes[1, 1].set_title('Classification Report: %s' % each_target)
    except:
        print('Error: could not plot classification results. Continuing...')

######################################################################################
def print_classification_model_stats(y_true, predicted, m_thresh=0.5):
    """
    This prints classification metrics in a nice format only for binary classes
    """
    # Use this to Test Classification Problems Only ####
    try:
        rare_class = find_rare_class(y_true)
        reg_acc = [0, 0]
        for i, threshold in zip(range(2), [0.5, m_thresh]):
            if threshold != 0.5:
                predicted[:, 0] = (predicted[:, 0] >= (1 - threshold)).astype('int')
                predicted[:, 1] = (predicted[:, 1] >= threshold).astype('int')
                y_pred = predicted[:, rare_class]
            else:
                y_pred = predicted.argmax(axis=1)
            print('Balanced Accuracy = %0.2f%% with Threshold = %0.2f' % (
                100 * balanced_accuracy_score(y_true, y_pred), threshold))
            print('Confusion Matrix:')
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            reg_acc[i] = balanced_accuracy_score(y_true, y_pred)
        print('#####################################################################')
        if reg_acc[0] >= reg_acc[1]:
            print('Regular threshold = %0.2f is better' % 0.5)
            return reg_acc[0]
        else:
            print('Modified threshold = %0.2f is better' % m_thresh)
            return reg_acc[1]
    except:
        print('Error: printing classification model metrics. Continuing...')
        return 0

################################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
def print_regression_model_stats(actuals, predicted, verbose=0):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    if isinstance(actuals,pd.Series) or isinstance(actuals,pd.DataFrame):
        actuals = actuals.values
    if isinstance(predicted,pd.Series) or isinstance(predicted,pd.DataFrame):
        predicted = predicted.values
    if len(actuals) != len(predicted):
        if verbose:
            print('Error: Number of rows in actuals and predicted dont match. Continuing...')
        return np.inf
    try:
        ### This is for Multi_Label Problems ###
        assert actuals.shape[1]
        multi_label = True
    except:
        multi_label = False
    if multi_label:
        for i in range(actuals.shape[1]):
            actuals_x = actuals[:,i]
            try:
                predicted_x = predicted[:,i]
            except:
                predicted_x = predicted[:]
            if verbose:
                print('for target %s:' %i)
            each_rmse = print_regression_metrics(actuals_x, predicted_x, verbose)
        final_rmse = np.mean(each_rmse)
    else:
        final_rmse = print_regression_metrics(actuals, predicted, verbose)
    return final_rmse
################################################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def MAPE(y_true, y_pred): 
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (array-like): The true (actual) values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The MAPE value, expressed as a percentage.

    Notes:
        - MAPE is a common metric for evaluating the accuracy of forecasting models.
        - The function handles potential division-by-zero errors by replacing zero values in `y_true` with 1, ensuring a stable calculation.
        - The formula used is: `mean(abs((y_true - y_pred) / max(1, abs(y_true)))) * 100`
    """    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100
################################################################################
def print_regression_metrics(y_true, y_preds, verbose=0):
    """
    Prints a comprehensive set of regression metrics to evaluate model performance.

    Args:
        y_true (array-like): The true (actual) values.
        y_preds (array-like): The predicted values.
        verbose (int, optional): Controls the level of output. If 1, prints detailed metrics. 
        If 0, prints a summary and generates a scatter plot. Defaults to 0.

    Returns:
        float: The Root Mean Squared Error (RMSE). Returns np.inf if an error occurs during calculation.

    Notes:
        - This function calculates and prints RMSE, Normalized RMSE, MAE, WAPE, Bias, MAPE, and R-Squared.
        - If `verbose` is set to 0, it generates a scatter plot of the true vs. predicted values using `plot_regression()`.
        - If there are zero values in `y_true`, it will print a warning that MAPE is not available and still calculates WAPE and Bias.
        - It handles potential exceptions during metric calculation and prints an error message if one occurs.
    """
    try:
        each_rmse = np.sqrt(mean_squared_error(y_true, y_preds))
        if verbose:
            print('    RMSE = %0.3f' %each_rmse)
            print('    Norm RMSE = %0.0f%%' %(100*np.sqrt(mean_squared_error(y_true, y_preds))/np.std(y_true)))
            print('    MAE = %0.3f'  %mean_absolute_error(y_true, y_preds))
        if len(y_true[(y_true==0)]) > 0:
            if verbose:
                print('    WAPE = %0.0f%%, Bias = %0.1f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                            100*np.sum(y_true-y_preds)/np.sum(y_true)))
                print('    No MAPE available since zeroes in actuals')
        else:
            if verbose:
                print('    WAPE = %0.0f%%, Bias = %0.1f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                            100*np.sum(y_true-y_preds)/np.sum(y_true)))
                mape = 100*MAPE(y_true, y_preds)
                print('    MAPE = %0.0f%%' %(mape))
                if mape > 100:
                    print('\tHint: high MAPE: try np.log(y) instead of (y).')
        print('    R-Squared = %0.0f%%' %(100*r2_score(y_true, y_preds)))
        if not verbose:
            plot_regression(y_true, y_preds, chart='scatter')
        return each_rmse
    except Exception as e:
        print('Could not print regression metrics due to %s.' %e)
        return np.inf
################################################################################
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev
################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    To avoid infinity due to division by zero, we select max(0.01, abs(actuals)) to show MAPE.
    """
    ### Wherever there is zero, replace it with 0.001 so it doesn't result in division by zero
    perc_err = (100*(np.where(y==0,0.001,y) - y_hat))/np.where(y==0,0.001,y)
    return np.mean(abs(perc_err))
    
def plot_regression(actuals, predicted, chart='scatter'):
    """
    This function plots the actuals vs. predicted as a line plot.
    You can change the chart type to "scatter' to get a scatter plot.
    """
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    plt.figure(figsize=figsize)
    if not isinstance(actuals, np.ndarray):
        actuals = actuals.values
    dfplot = pd.DataFrame([actuals,predicted]).T
    dfplot.columns = ['Actual','Forecast']
    dfplot = dfplot.sort_index()
    lineStart = actuals.min()
    lineEnd = actuals.max()
    if chart == 'line':
        plt.plot(dfplot)
    else:
        plt.scatter(actuals, predicted, color = next(colors), alpha=0.5,label='Predictions')
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
        plt.xlim(lineStart, lineEnd)
        plt.ylim(lineStart, lineEnd)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Model: Predicted vs Actuals', fontsize=12)
    plt.show();
###########################################################################
from sklearn.metrics import roc_auc_score
import copy
from sklearn.metrics import balanced_accuracy_score, classification_report
import pdb
def print_sulo_accuracy(y_test, y_preds, y_probas='', verbose=0):
    """
    A wrapper function for print_classification_metrics,  meant for compatibility with older featurewiz versions.
    Usage:
    print_sulo_accuracy(y_test, y_preds, y_probas, verbose-0)
    """
    return print_classification_metrics(y_test, y_preds, y_probas, verbose)

def print_classification_metrics(y_test, y_preds, y_probas='', verbose=0):
    """
    Calculate and print classification metrics for single-label, multi-label, and multi-class problems.

    This function computes and displays various metrics such as balanced accuracy score and ROC AUC score 
    based on the given test labels, predicted labels, and predicted probabilities. It handles different 
    scenarios including single-label, multi-label, multi-class, and their combinations. Additionally, it 
    provides detailed classification reports if verbose output is requested.

    Parameters:
    y_test (array-like): True labels. Should be 1D for single-label and 2D for multi-label problems.
    y_preds (array-like): Predicted labels. Should match the dimensionality of y_test.
    y_probas (array-like, optional): Predicted probabilities. Default is an empty string, indicating 
                                     no probabilities are provided. Should be 2D with probabilities for 
                                     each class.
    verbose (int, optional): Verbose level. If set to 1, it prints out detailed classification reports. 
                             Default is 0, which prints only summary metrics.

    Returns:
    float: Final average balanced accuracy score across all labels/classes. Returns 0.0 if an exception occurs.

    Raises:
    Exception: If an error occurs during the calculation or printing of metrics.

    Note:
    The function is designed to handle various edge cases and different formats of predicted probabilities, 
    such as those produced by different classifiers or methods like Label Propagation.

    Examples:
    # For single-label binary classification
    print_classification_metrics(y_test, y_preds)

    # For multi-label classification with verbose output
    print_classification_metrics(y_test, y_preds, verbose=1)

    # For single-label classification with predicted probabilities
    print_classification_metrics(y_test, y_preds, y_probas)
    """    
    try:
        bal_scores = []
        ####### Once you have detected what problem it is, now print its scores #####
        if y_test.ndim <= 1: 
            ### This is a single label problem # we need to test for multiclass ##
            bal_score = balanced_accuracy_score(y_test,y_preds)
            print('Bal accu %0.0f%%' %(100*bal_score))
            if not isinstance(y_probas, str):
                if y_probas.ndim <= 1:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                else:
                    if y_probas.shape[1] == 2:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                    else:
                        print('Multi-class ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
            bal_scores.append(bal_score)
            if verbose:
                print(classification_report(y_test,y_preds))
        elif y_test.ndim >= 2:
            if y_test.shape[1] == 1:
                bal_score = balanced_accuracy_score(y_test,y_preds)
                bal_scores.append(bal_score)
                print('Bal accu %0.0f%%' %(100*bal_score))
                if not isinstance(y_probas, str):
                    if y_probas.shape[1] > 2:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
                    else:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                if verbose:
                    print(classification_report(y_test,y_preds))
            else:
                if isinstance(y_probas, str):
                    ### This is for multi-label problems without probas ####
                    for each_i in range(y_test.shape[1]):
                        bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
                        bal_scores.append(bal_score)
                        print('    Bal accu %0.0f%%' %(100*bal_score))
                        if verbose:
                            print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
                else:
                    ##### This is only for multi_label_multi_class problems
                    num_targets = y_test.shape[1]
                    for each_i in range(num_targets):
                        print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])))
                        if len(np.unique(y_test.values[:,each_i])) > 2:
                            ### This nan problem happens due to Label Propagation but can be fixed as follows ##
                            mat = y_probas[each_i]
                            if np.any(np.isnan(mat)):
                                mat = pd.DataFrame(mat).fillna(method='ffill').values
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                            else:
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                        else:
                            if isinstance(y_probas, dict):
                                if y_probas[each_i].ndim <= 1:
                                    ## This is caused by Label Propagation hence you must probas like this ##
                                    mat = y_probas[each_i]
                                    if np.any(np.isnan(mat)):
                                        mat = pd.DataFrame(mat).fillna(method='ffill').values
                                    bal_score = roc_auc_score(y_test.values[:,each_i],mat)
                                else:
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i][:,1])
                            else:
                                if y_probas.shape[1] == num_targets:
                                    ### This means Label Propagation was used which creates probas like this ##
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[:,each_i])
                                else:
                                    ### This means regular sklearn classifiers which predict multi dim probas #
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i])
                        print('Target number %s: ROC AUC score %0.0f%%' %(each_i+1,100*bal_score))
                        bal_scores.append(bal_score)
                        if verbose:
                            print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
        final_score = np.mean(bal_scores)
        if verbose:
            print("final average balanced accuracy score = %0.2f" %final_score)
        return final_score
    except Exception as e:
        print('Could not print classification metrics due to %s.' %e)
        return 0.0
######################################################################################################


