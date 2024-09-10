import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, roc_auc_score

def plot_categories(df, features, target):

    num_cols = min(2, len(features))
    num_rows = (len(features) - 1) // 2 + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 3*num_rows))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    colors = ['tab:red', 'tab:green']

    for i, column in enumerate(features):
        row_idx = i // num_cols
        col_idx = i % num_cols
        sns.countplot(x=column, data=df, hue=target, ax=axes[row_idx, col_idx], palette=colors, saturation=0.5, width=0.75)
        axes[row_idx, col_idx].set_xlabel(column)
        axes[row_idx, col_idx].set_ylabel('Count')
        axes[row_idx, col_idx].legend(title=target, loc='upper right', facecolor='white')

    for i in range(len(features), num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()




def plot_top_n_cities(df, feature, target, n=10):

    top_n = df[feature].value_counts().nlargest(n).index
    df_top_n = df[df[feature].isin(top_n)]
    
    city_counts = df_top_n[feature].value_counts()
    
    sorted_cities = city_counts.index
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['tab:red', 'tab:green']
    
    sns.countplot(x=feature, data=df_top_n, hue=target, ax=ax, palette=colors, saturation=0.5, width=0.75, order=sorted_cities)
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(f'Top {n} cities based on the number of reviews')
    ax.legend(title=target, loc='upper right', facecolor='white')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




def plot_top_n_cities_by_pct_target(df, feature, target, target_value=None, n=10):

    city_counts = df[feature].value_counts()
    target_counts = df.groupby(feature)[target].value_counts().unstack(fill_value=0)
    
    percentages = target_counts.div(city_counts, axis=0) * 100
    
    top_n_cities = percentages[target_value].nlargest(n).index
    df_top_n = df[df[feature].isin(top_n_cities)]
    
    top_n_percentages = percentages.loc[top_n_cities]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = [0] * len(top_n_cities)
    
    for target_category in percentages.columns:
        ax.bar(top_n_percentages.index, top_n_percentages[target_category], label=target_category,
               bottom=bottom, color='#aa5353' if target_category == 'negative' else '#498349')
        bottom = [b + h for b, h in zip(bottom, top_n_percentages[target_category])]

    ax.set_xlabel(feature)
    ax.set_ylabel('Percentage')
    ax.set_title(f'Top {n} cities with the highest percentage of {target_value} reviews')
    ax.legend(title=target, loc='upper right', facecolor='white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def wordcloud_by_class(df, text_feature, target, value):

    all_reviews = " ".join(review for review in df.loc[df[target] == value][text_feature])

    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='tab10',
                      max_words=100, collocations=False, random_state=42).generate(all_reviews)

    plt.figure(figsize=(9, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()




def plot_roc_curve(models_dict, y_test):

    colors = ['tab:blue','tab:red','tab:orange','tab:green','tab:purple','tab:gray','tab:cyan','tab:pink']
    plt.figure(figsize=(9, 6))
    for (model_name, model_details), color in zip(models_dict.items(), colors):
        y_proba = model_details[0].predict_proba(model_details[1])[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--', label='Random Model')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.show()