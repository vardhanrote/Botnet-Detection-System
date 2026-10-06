import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    load_data,
    clean_data,
    build_feature_sets,
    ridge_selection,
    scale_features,
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

RANDOM_STATE = 42

PROCESSED_DIR = "data/processed"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    train_df, test_df = load_data()


    # ==================================================
    # 2. SAVE TARGETS
    # ==================================================

    y_train_binary = train_df["label"].copy()
    y_test_binary = test_df["label"].copy()

    y_train_multiclass = train_df[
        "attack_cat"
    ].copy()

    y_test_multiclass = test_df[
        "attack_cat"
    ].copy()


    # ==================================================
    # 3. CLEAN DATA
    # ==================================================

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)


    # ==================================================
    # 4. CREATE NETWORK + DNS FEATURES
    # ==================================================

    (
        network_train,
        network_test,
        dns_train,
        dns_test
    ) = build_feature_sets(
        train_df,
        test_df
    )


    # ==================================================
    # 5. RIDGE FEATURE SELECTION
    # ==================================================

    print(
        "\n========== NETWORK FEATURE SELECTION =========="
    )

    selected_network, network_ranking = ridge_selection(
        network_train,
        y_train_binary,
        number_of_features=5
    )

    print("\nNetwork ranking:")

    print(network_ranking)

    print(
        "\nSelected network features:"
    )

    print(selected_network)


    print(
        "\n========== DNS FEATURE SELECTION =========="
    )

    selected_dns, dns_ranking = ridge_selection(
        dns_train,
        y_train_binary,
        number_of_features=5
    )

    print("\nDNS ranking:")

    print(dns_ranking)

    print(
        "\nSelected DNS features:"
    )

    print(selected_dns)


    # ==================================================
    # 6. KEEP SELECTED FEATURES
    # ==================================================

    network_train = network_train[
        selected_network
    ]

    network_test = network_test[
        selected_network
    ]

    dns_train = dns_train[
        selected_dns
    ]

    dns_test = dns_test[
        selected_dns
    ]


    # ==================================================
    # 7. SCALE FEATURES
    # ==================================================

    print(
        "\n========== SCALING FEATURES =========="
    )

    (
        network_train,
        network_test,
        network_scaler
    ) = scale_features(
        network_train,
        network_test
    )

    (
        dns_train,
        dns_test,
        dns_scaler
    ) = scale_features(
        dns_train,
        dns_test
    )


    # ==================================================
    # 8. COMBINE TWO STREAMS TEMPORARILY
    # ==================================================
    #
    # We combine them only for SMOTE.
    #
    # This is important because network and DNS rows
    # must remain paired for the neural network.
    #
    # After SMOTE we split them again.
    # ==================================================

    print(
        "\n========== PREPARING SMOTE =========="
    )

    combined_train = pd.concat(
        [
            network_train.reset_index(drop=True),
            dns_train.reset_index(drop=True)
        ],
        axis=1
    )

    print(
        "Combined training shape:",
        combined_train.shape
    )


    # ==================================================
    # 9. MULTICLASS TARGET ENCODING
    # ==================================================

    class_names = sorted(
        y_train_multiclass.unique()
    )

    class_to_id = {
        name: index
        for index, name in enumerate(class_names)
    }

    id_to_class = {
        index: name
        for name, index in class_to_id.items()
    }

    y_train_multi_encoded = (
        y_train_multiclass
        .map(class_to_id)
        .astype(int)
        .values
    )

    y_test_multi_encoded = (
        y_test_multiclass
        .map(class_to_id)
        .astype(int)
        .values
    )


    # ==================================================
    # 10. APPLY SMOTE
    # ==================================================
    #
    # IMPORTANT:
    # SMOTE is applied ONLY to the training data.
    #
    # We use the multiclass labels because the final
    # model has a 10-class output.
    # ==================================================

    from imblearn.over_sampling import SMOTE

    print(
        "\n========== CLASS DISTRIBUTION BEFORE SMOTE =========="
    )

    print(
        pd.Series(
            y_train_multi_encoded
        ).value_counts().sort_index()
    )


    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    combined_resampled, y_train_resampled = (
        smote.fit_resample(
            combined_train,
            y_train_multi_encoded
        )
    )


    print(
        "\n========== CLASS DISTRIBUTION AFTER SMOTE =========="
    )

    print(
        pd.Series(
            y_train_resampled
        ).value_counts().sort_index()
    )


    # ==================================================
    # 11. SPLIT NETWORK + DNS AGAIN
    # ==================================================

    network_columns = [
        f"network_{feature}"
        for feature in selected_network
    ]

    dns_columns = [
        f"dns_{feature}"
        for feature in selected_dns
    ]


    # Rename combined columns first
    combined_resampled.columns = (
        network_columns +
        dns_columns
    )


    network_train_resampled = (
        combined_resampled[
            network_columns
        ]
    )

    dns_train_resampled = (
        combined_resampled[
            dns_columns
        ]
    )


    # ==================================================
    # 12. CONVERT TO NUMPY ARRAYS
    # ==================================================

    X_network_train = (
        network_train_resampled
        .values
        .astype(np.float32)
    )

    X_dns_train = (
        dns_train_resampled
        .values
        .astype(np.float32)
    )

    X_network_test = (
        network_test
        .values
        .astype(np.float32)
    )

    X_dns_test = (
        dns_test
        .values
        .astype(np.float32)
    )


    y_binary_train = (
        np.array(y_train_resampled > 0)
        .astype(np.float32)
    )

    y_binary_test = (
        y_test_binary
        .values
        .astype(np.float32)
    )

    y_multi_train = (
        np.array(y_train_resampled)
        .astype(np.int64)
    )

    y_multi_test = (
        y_test_multi_encoded
        .astype(np.int64)
    )


    # ==================================================
    # 13. SAVE ARRAYS
    # ==================================================

    print(
        "\n========== SAVING MODEL-READY DATA =========="
    )

    np.save(
        f"{PROCESSED_DIR}/X_network_train.npy",
        X_network_train
    )

    np.save(
        f"{PROCESSED_DIR}/X_dns_train.npy",
        X_dns_train
    )

    np.save(
        f"{PROCESSED_DIR}/X_network_test.npy",
        X_network_test
    )

    np.save(
        f"{PROCESSED_DIR}/X_dns_test.npy",
        X_dns_test
    )

    np.save(
        f"{PROCESSED_DIR}/y_binary_train.npy",
        y_binary_train
    )

    np.save(
        f"{PROCESSED_DIR}/y_binary_test.npy",
        y_binary_test
    )

    np.save(
        f"{PROCESSED_DIR}/y_multiclass_train.npy",
        y_multi_train
    )

    np.save(
        f"{PROCESSED_DIR}/y_multiclass_test.npy",
        y_multi_test
    )


    # ==================================================
    # 14. SAVE FEATURE INFORMATION
    # ==================================================

    feature_information = {
        "network_features": selected_network,
        "dns_features": selected_dns,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
    }

    joblib.dump(
        feature_information,
        f"{PROCESSED_DIR}/feature_information.pkl"
    )

    joblib.dump(
        network_scaler,
        f"{PROCESSED_DIR}/network_scaler.pkl"
    )

    joblib.dump(
        dns_scaler,
        f"{PROCESSED_DIR}/dns_scaler.pkl"
    )


    # ==================================================
    # 15. FINAL SHAPES
    # ==================================================

    print(
        "\n========== FINAL MODEL-READY SHAPES =========="
    )

    print(
        "Network train:",
        X_network_train.shape
    )

    print(
        "DNS train:",
        X_dns_train.shape
    )

    print(
        "Binary train:",
        y_binary_train.shape
    )

    print(
        "Multiclass train:",
        y_multi_train.shape
    )

    print(
        "\nNetwork test:",
        X_network_test.shape
    )

    print(
        "DNS test:",
        X_dns_test.shape
    )

    print(
        "Binary test:",
        y_binary_test.shape
    )

    print(
        "Multiclass test:",
        y_multi_test.shape
    )


    print(
        "\n========== PREPROCESSING COMPLETE =========="
    )


if __name__ == "__main__":
    main()