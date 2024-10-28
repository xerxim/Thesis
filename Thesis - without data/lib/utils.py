import numpy as np
import pandas as pd
import re
import statsmodels.formula.api as smf


def clean_cats(df, cols):
    """
    Clean the categories of columns.
    Replaces negative values with nan and deletes text from other int values.

    :param df: A pandas dataframe.
    :param cols: Name(s) of columns to clean up.
    :return: Cleaned dataframe.
    """
    if not isinstance(cols, list):
        cols = [cols]
    for col in cols:
        # Get categories for column.
        try:
            cats = df[col].cat.categories
        except:
            print(f"Column {col} is probably not categorical.")
            continue
        # Iterate over categories.
        rename_dict = {}
        regex = "[^-\d]*(-?\d+).*"
        for cat in cats:
            if isinstance(cat, str):
                # Extract int.
                el = int(re.search(regex, cat).group(1))
                # Make nan if negative.
                if el < 0:
                    el = np.nan
                # Add to dict
                rename_dict[cat] = el
        # Rename categories.
        if len(rename_dict) > 0:
            df[col] = df[col].replace(rename_dict)

    return df


def classify(row: pd.Series) -> str:
    """
    Assigns relationship type by value combinations.
    This approach uses four categories:
    trad/trad, egal/egal, fegal/mtrad, ftrad/megal

    :param row: Row of dataframe with male and female values.
    :return: Category as string.
    """
    if row["ftrad"] and row["mtrad"]:
        return "trad/trad"
    if (row["fegal"]) and (row["megal"]):
        return "egal/egal"
    if row["fegal"] and (row["mtrad"]):
        return "fegal/mtrad"
    else:
        return "ftrad/megal"


def mixed(row: pd.Series) -> str:
    """
    Assigns relationship type by value combinations.
    This is an alternative approach using a mixed category.

    :param row: Row of dataframe with male and female values.
    :return: Category as string.
    """
    if row["ftrad"] and row["mtrad"]:
        return "trad/trad"
    if (row["fegal"]) and (row["megal"]):
        return "egal/egal"
    else:
        return "mixed"


def stepwise(
        dependent: str,
        independents: [str],
        data: pd.DataFrame
):
    """
    Computes a stepwise linear regression with multiple variables,
    adding them one at a time.

    :param dependent:    Name of dependent variable.
    :param independents: List of independent variables in order
                         of steps.
    :param data:         Dataframe on which regression takes place.
    """
    model = f"{dependent} ~ {independents[0]}"
    ols = smf.ols(model, data=data).fit()
    print("Step 0:")
    print(ols.summary())
    for i, cv in enumerate(independents[1:]):
        model = f"{model} + {cv}"
        ols = smf.ols(model, data=data).fit()
        print(f"Step {i + 1}:")
        print(ols.summary())
