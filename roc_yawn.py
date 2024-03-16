import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data():
    df = pd.read_csv('yawn_data.csv')
    return df

def one_rule_threshold(df):
    average_mar_values = df['Average_MAR']
    best_threshold = 0
    min_val = min(average_mar_values)
    max_val = max(average_mar_values)
    gap = 0.025
    min_mistakes = float('inf')
    MAR_range = np.arange(min_val, max_val + gap, gap)
    total_fpr = []
    total_tpr = []
    for threshold in MAR_range:
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0
        for index, val in enumerate(average_mar_values):
            if val > threshold and df['Yawn'].iloc[index] == 0:
                false_negative += 1

            elif val <= threshold and df['Yawn'].iloc[index] == 1:
                false_positive += 1
            
            elif val < threshold and df['Yawn'].iloc[index] == 0:
                true_negative += 1
            elif val >= threshold and df['Yawn'].iloc[index] == 1:
                true_positive += 1

        # Add a check to avoid division by zero
        if (true_positive + false_negative) > 0:
            curr_tpr = true_positive / (true_positive + false_negative)
            total_tpr.append(curr_tpr)
        else:
            total_tpr.append(0)

        if (false_positive + true_negative) > 0:
            curr_fpr = false_positive / (false_positive + true_negative)
            total_fpr.append(curr_fpr)
        else:
            total_fpr.append(0)

        curr_total_mistakes = false_negative + false_positive
        
        if curr_total_mistakes < min_mistakes:
            min_mistakes = curr_total_mistakes
            best_threshold = threshold

    return best_threshold, total_tpr, total_fpr

def main():
    df = read_data()

    # Get predicted labels using the one-rule threshold approach
    best_threshold, total_tpr, total_fpr = one_rule_threshold(df)

    # Plot ROC curve
    plt.figure()
    plt.plot(total_fpr, total_tpr, marker='o', color='darkorange', label='Custom ROC point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Custom ROC Point')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()