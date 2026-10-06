import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def ridge_feature_selection(
    X,
    y,
    n_features=10
):
    """
    Select important features using Ridge Regression.

    The absolute value of the Ridge coefficient
    is used as the importance score.
    """

    # Scale features before Ridge
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Create Ridge model
    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_scaled,
        y
    )

    # Calculate feature importance
    importance = np.abs(
        ridge.coef_
    )

    # Get feature names
    feature_names = X.columns

    # Create ranking
    ranking = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top features
    selected_features = [
        feature
        for feature, score in ranking[:n_features]
    ]

    print("\n========== RIDGE FEATURE RANKING ==========")

    for rank, (feature, score) in enumerate(
        ranking,
        start=1
    ):
        print(
            f"{rank}. {feature}: {score:.4f}"
        )

    print(
        f"\nSelected top {n_features} features:"
    )

    print(selected_features)

    return selected_features, ranking