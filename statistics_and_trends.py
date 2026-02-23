import sys

if 'google.colab' in sys.modules:
    !pip install corner

"""
This is the template file for the statistics and trends assignment.

You are expected to complete all sections and make this a fully working,
documented file.

IMPORTANT:
- Do NOT change any function, file, or variable names if given here.
- Keep code PEP-8 compliant (including docstrings).
- The template expects to read from 'data.csv' and save plots as PNG files:
  relational_plot.png, categorical_plot.png, statistical_plot.png
"""

from corner import corner 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Create a relational plot (scatter) and save it as 'relational_plot.png'.

    Chosen plot:
    - Scatter plot of Age vs Fare (Titanic dataset).
    """
    fig, ax = plt.subplots()

    # Scatter: Age vs Fare
    ax.scatter(df["Age"], df["Fare"])
    ax.set_title("Relational Plot: Fare vs Age (Titanic)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")

    fig.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """
    Create a categorical plot (bar) and save it as 'categorical_plot.png'.

    Chosen plot:
    - Bar chart of survival counts by gender.
    """
    # Count Survived (0/1) grouped by Sex
    counts = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Categorical Plot: Survival Counts by Gender (Titanic)")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Passengers")
    ax.legend(title="Survived", labels=["No (0)", "Yes (1)"])

    fig.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close(fig)
    return


def plot_statistical_plot(df):
    """
    Create a statistical plot (correlation heatmap) and save it as
    'statistical_plot.png'.
    """
    # Select numeric columns for correlation analysis
    numeric_cols = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex_num",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Statistical Plot: Correlation Heatmap (Titanic)")

    fig.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close(fig)
    return


def statistical_analysis(df, col: str):
    """
    Compute the four key statistical moments for a chosen numeric column:
    - mean (1st moment)
    - standard deviation (2nd moment)
    - skewness (3rd moment)
    - excess kurtosis (4th moment)

    Returns:
        (mean, stddev, skew, excess_kurtosis)
    """
    series = df[col].dropna().astype(float)

    mean = float(series.mean())
    stddev = float(series.std(ddof=1))
    skewness = float(ss.skew(series, bias=False))
    # fisher=True => excess kurtosis (0 means normal-like)
    excess_kurtosis = float(ss.kurtosis(series, fisher=True, bias=False))

    return mean, stddev, skewness, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the Titanic dataset and demonstrate quick exploration features:
    - head()
    - describe()
    - corr()

    Cleaning/Preparation steps:
    - Drop duplicates
    - Drop Cabin (high missingness)
    - Fill missing Age with median
    - Fill missing Embarked with mode
    - Encode Sex (male=0, female=1) into Sex_num
    - One-hot encode Embarked into Embarked_C/Embarked_Q/Embarked_S
    """
    # Quick exploration outputs
    print("First 5 rows:\n", df.head())
    print("\nDescribe (numeric):\n", df.describe(include=np.number))
    print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))

    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"\nDuplicates removed: {before - after}")

    # Drop Cabin if present
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])

    # Fill missing Age with median
    if "Age" in df.columns and df["Age"].isnull().any():
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing Embarked with mode
    if "Embarked" in df.columns and df["Embarked"].isnull().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode(dropna=True)[0])

    # Encode Sex to numeric values
    if "Sex" in df.columns:
        df["Sex_num"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot encode Embarked
    if "Embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=False)
        df = pd.concat([df, embarked_dummies], axis=1)

    # Correlation preview 
    numeric_preview = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex_num",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ]
    cols = [c for c in numeric_preview if c in df.columns]
    print("\nCorrelation (selected numeric columns):\n", df[cols].corr())

    # Final missing check
    print("\nTotal remaining missing values:", int(df.isnull().sum().sum()))
    return df


def writing(moments, col):
    """
    Print the summary of statistical moments and a short distribution statement.
    """
    mean, stddev, skewness, excess_kurtosis = moments

    print(f"\nFor the attribute {col}:")
    print(
        f"Mean = {mean:.2f}, "
        f"Standard Deviation = {stddev:.2f}, "
        f"Skewness = {skewness:.2f}, and "
        f"Excess Kurtosis = {excess_kurtosis:.2f}."
    )

    # Skewness interpretation
    if skewness > 0.5:
        skew_text = "right skewed"
    elif skewness < -0.5:
        skew_text = "left skewed"
    else:
        skew_text = "not skewed"

    # Kurtosis interpretation
    if excess_kurtosis > 1:
        kurt_text = "leptokurtic"
    elif excess_kurtosis < -1:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"

    print(f"The data was {skew_text} and {kurt_text}.")
    return


def main():
    """
    Main execution:
    - Reads 'data.csv'
    - Preprocesses data
    - Generates three plots (saved as PNG)
    - Computes statistical moments for a chosen column
    - Prints findings
    """
    df = pd.read_csv("Titanic-Dataset.csv")
    df = preprocessing(df)

    # Chosen column for analysis (numeric, informative distribution)
    col = "Fare"

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == "__main__":
    main()
