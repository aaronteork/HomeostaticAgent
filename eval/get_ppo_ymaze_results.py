import ast
from pathlib import Path

import pandas as pd


def get_results(path):
    df = pd.read_csv(path)

    # 1. Vectorize the parsing of the list column
    # Using .str accessor is much faster than .apply(ast.literal_eval)
    df["resources_consumed"] = df["resources_consumed"].apply(ast.literal_eval)
    resources = pd.DataFrame(df["resources_consumed"].tolist())

    df["first_resource_consumed"] = resources.get(0)
    df["second_resource_consumed"] = resources.get(1)

    # 2. Vectorized boolean logic for 'correct' resources
    # This replaces the row-wise apply with simple comparisons
    hunger_less_than_thirst = df["initial_hunger"] < df["initial_thirst"]

    df["correct_first_resource"] = hunger_less_than_thirst.map(
        {True: "food", False: "water"}
    )
    df["correct_second_resource"] = hunger_less_than_thirst.map(
        {True: "water", False: "food"}
    )

    # 3. Direct vectorized comparison
    df["success_first"] = df["first_resource_consumed"] == df["correct_first_resource"]
    df["success_second"] = (
        df["second_resource_consumed"] == df["correct_second_resource"]
    )
    df["success"] = df["success_first"] & df["success_second"]

    # 4. Use meaningful filtering for assertions
    # Boolean indexing is much more readable than .loc with explicit False comparisons
    invalid_cases = df[~df["success_first"] & df["success_second"]]
    assert invalid_cases.empty, (
        "Found cases with correct second resource but incorrect first."
    )

    # Print results
    assert df.loc[df["success_second"] & ~df["success_first"]].empty, "Success second but fail first should not exist."

    # First resource
    print(f"Consumed the first resource correctly in {df["success_first"].sum()}/100 of episodes.")
    print(f"Consumed the wrong resource in the first instance in {df["first_resource_consumed"].ne(df["correct_first_resource"]).sum()}/100 of episodes.")
    print(f"Didnt consume anything in the first instance in {df['first_resource_consumed'].eq('nothing').sum()}/100 of episodes.")

    # Second resource
    df_temp = df.loc[df['success_first']]
    correct_episodes = df_temp['success_first'].sum()
    print(f"Consumed the second resource correctly in {df_temp['success_second'].sum()}/{correct_episodes} of episodes.")
    print(f"Consumed the wrong resource in the second instance in {df_temp['second_resource_consumed'].ne(df_temp['correct_second_resource']).sum()}/{correct_episodes} of episodes.")
    print(f"Didnt consume anything in the second instance in {df_temp['second_resource_consumed'].eq('nothing').sum()}/{correct_episodes} of episodes.")

    return


if __name__ == "__main__":
    path = (
        Path(__file__).resolve().parent
        / "ppo"
        / "Front_final"
        / "ppo_ymaze_episode_stats.csv"
    )
    results = get_results(path)
