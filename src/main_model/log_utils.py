from __future__ import annotations

import csv
import numpy as np


def save_training_log(
    *,
    log_path: str,
    log_csv_path: str,
    train_loss_history: list[float],
    val_loss_history: list[float],
    train_mae_history: list[float],
    val_mae_history: list[float],
    train_rmse_history: list[float],
    val_rmse_history: list[float],
    train_r2_history: list[float],
    val_r2_history: list[float],
    train_task_loss_history: list[np.ndarray],
    val_task_loss_history: list[np.ndarray],
    train_task_mae_history: list[np.ndarray],
    val_task_mae_history: list[np.ndarray],
    train_task_rmse_history: list[np.ndarray],
    val_task_rmse_history: list[np.ndarray],
    train_task_r2_history: list[np.ndarray],
    val_task_r2_history: list[np.ndarray],
    best_epoch: int,
    train_total_rows_before: int | None = None,
    train_kept_rows: int | None = None,
    train_dropped_total_rows: int | None = None,
    train_solute_skipped_rows: int | None = None,
    train_solute_invalid_count: int | None = None,
    train_solute_missing_count: int | None = None,
    train_solvent_smiles_required: int | None = None,
    train_solvent_skipped_rows: int | None = None,
    train_solvent_invalid_count: int | None = None,
    train_solvent_missing_count: int | None = None,
    val_total_rows_before: int | None = None,
    val_kept_rows: int | None = None,
    val_dropped_total_rows: int | None = None,
    val_solute_skipped_rows: int | None = None,
    val_solute_invalid_count: int | None = None,
    val_solute_missing_count: int | None = None,
    val_solvent_smiles_required: int | None = None,
    val_solvent_skipped_rows: int | None = None,
    val_solvent_invalid_count: int | None = None,
    val_solvent_missing_count: int | None = None,
    test_total_rows_before: int | None = None,
    test_kept_rows: int | None = None,
    test_dropped_total_rows: int | None = None,
    test_solute_skipped_rows: int | None = None,
    test_solute_invalid_count: int | None = None,
    test_solute_missing_count: int | None = None,
    test_solvent_smiles_required: int | None = None,
    test_solvent_skipped_rows: int | None = None,
    test_solvent_invalid_count: int | None = None,
    test_solvent_missing_count: int | None = None,
) -> None:
    train_loss_arr = np.array(train_loss_history, dtype=float)
    val_loss_arr = np.array(val_loss_history, dtype=float)
    train_mae_arr = np.array(train_mae_history, dtype=float)
    val_mae_arr = np.array(val_mae_history, dtype=float)
    train_rmse_arr = np.array(train_rmse_history, dtype=float)
    val_rmse_arr = np.array(val_rmse_history, dtype=float)
    train_r2_arr = np.array(train_r2_history, dtype=float)
    val_r2_arr = np.array(val_r2_history, dtype=float)
    train_task_loss_arr = np.array(train_task_loss_history, dtype=float)
    val_task_loss_arr = np.array(val_task_loss_history, dtype=float)
    train_task_mae_arr = np.array(train_task_mae_history, dtype=float)
    val_task_mae_arr = np.array(val_task_mae_history, dtype=float)
    train_task_rmse_arr = np.array(train_task_rmse_history, dtype=float)
    val_task_rmse_arr = np.array(val_task_rmse_history, dtype=float)
    train_task_r2_arr = np.array(train_task_r2_history, dtype=float)
    val_task_r2_arr = np.array(val_task_r2_history, dtype=float)

    def _scalar_arr(x):
        return np.array([np.nan if x is None else x], dtype=float)

    train_total_rows_before_arr = _scalar_arr(train_total_rows_before)
    train_kept_rows_arr = _scalar_arr(train_kept_rows)
    train_dropped_total_rows_arr = _scalar_arr(train_dropped_total_rows)
    train_solute_skipped_rows_arr = _scalar_arr(train_solute_skipped_rows)
    train_solute_invalid_count_arr = _scalar_arr(train_solute_invalid_count)
    train_solute_missing_count_arr = _scalar_arr(train_solute_missing_count)
    train_solvent_smiles_required_arr = _scalar_arr(train_solvent_smiles_required)
    train_solvent_skipped_rows_arr = _scalar_arr(train_solvent_skipped_rows)
    train_solvent_invalid_count_arr = _scalar_arr(train_solvent_invalid_count)
    train_solvent_missing_count_arr = _scalar_arr(train_solvent_missing_count)

    val_total_rows_before_arr = _scalar_arr(val_total_rows_before)
    val_kept_rows_arr = _scalar_arr(val_kept_rows)
    val_dropped_total_rows_arr = _scalar_arr(val_dropped_total_rows)
    val_solute_skipped_rows_arr = _scalar_arr(val_solute_skipped_rows)
    val_solute_invalid_count_arr = _scalar_arr(val_solute_invalid_count)
    val_solute_missing_count_arr = _scalar_arr(val_solute_missing_count)
    val_solvent_smiles_required_arr = _scalar_arr(val_solvent_smiles_required)
    val_solvent_skipped_rows_arr = _scalar_arr(val_solvent_skipped_rows)
    val_solvent_invalid_count_arr = _scalar_arr(val_solvent_invalid_count)
    val_solvent_missing_count_arr = _scalar_arr(val_solvent_missing_count)

    test_total_rows_before_arr = _scalar_arr(test_total_rows_before)
    test_kept_rows_arr = _scalar_arr(test_kept_rows)
    test_dropped_total_rows_arr = _scalar_arr(test_dropped_total_rows)
    test_solute_skipped_rows_arr = _scalar_arr(test_solute_skipped_rows)
    test_solute_invalid_count_arr = _scalar_arr(test_solute_invalid_count)
    test_solute_missing_count_arr = _scalar_arr(test_solute_missing_count)
    test_solvent_smiles_required_arr = _scalar_arr(test_solvent_smiles_required)
    test_solvent_skipped_rows_arr = _scalar_arr(test_solvent_skipped_rows)
    test_solvent_invalid_count_arr = _scalar_arr(test_solvent_invalid_count)
    test_solvent_missing_count_arr = _scalar_arr(test_solvent_missing_count)

    np.savez(
        log_path,
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
        train_mae=train_mae_arr,
        val_mae=val_mae_arr,
        train_rmse=train_rmse_arr,
        val_rmse=val_rmse_arr,
        train_r2=train_r2_arr,
        val_r2=val_r2_arr,
        train_task_loss=train_task_loss_arr,
        val_task_loss=val_task_loss_arr,
        train_task_mae=train_task_mae_arr,
        val_task_mae=val_task_mae_arr,
        train_task_rmse=train_task_rmse_arr,
        val_task_rmse=val_task_rmse_arr,
        train_task_r2=train_task_r2_arr,
        val_task_r2=val_task_r2_arr,
        best_epoch=np.array([best_epoch], dtype=int),
        train_total_rows_before=train_total_rows_before_arr,
        train_kept_rows=train_kept_rows_arr,
        train_dropped_total_rows=train_dropped_total_rows_arr,
        train_solute_skipped_rows=train_solute_skipped_rows_arr,
        train_solute_invalid_count=train_solute_invalid_count_arr,
        train_solute_missing_count=train_solute_missing_count_arr,
        train_solvent_smiles_required=train_solvent_smiles_required_arr,
        train_solvent_skipped_rows=train_solvent_skipped_rows_arr,
        train_solvent_invalid_count=train_solvent_invalid_count_arr,
        train_solvent_missing_count=train_solvent_missing_count_arr,
        val_total_rows_before=val_total_rows_before_arr,
        val_kept_rows=val_kept_rows_arr,
        val_dropped_total_rows=val_dropped_total_rows_arr,
        val_solute_skipped_rows=val_solute_skipped_rows_arr,
        val_solute_invalid_count=val_solute_invalid_count_arr,
        val_solute_missing_count=val_solute_missing_count_arr,
        val_solvent_smiles_required=val_solvent_smiles_required_arr,
        val_solvent_skipped_rows=val_solvent_skipped_rows_arr,
        val_solvent_invalid_count=val_solvent_invalid_count_arr,
        val_solvent_missing_count=val_solvent_missing_count_arr,
        test_total_rows_before=test_total_rows_before_arr,
        test_kept_rows=test_kept_rows_arr,
        test_dropped_total_rows=test_dropped_total_rows_arr,
        test_solute_skipped_rows=test_solute_skipped_rows_arr,
        test_solute_invalid_count=test_solute_invalid_count_arr,
        test_solute_missing_count=test_solute_missing_count_arr,
        test_solvent_smiles_required=test_solvent_smiles_required_arr,
        test_solvent_skipped_rows=test_solvent_skipped_rows_arr,
        test_solvent_invalid_count=test_solvent_invalid_count_arr,
        test_solvent_missing_count=test_solvent_missing_count_arr,
    )
    print(f"[Log] saved -> {log_path}")

    with open(log_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = [
            "epoch",
            "is_best_epoch",
            "train_loss",
            "val_loss",
            "train_mae",
            "val_mae",
            "train_rmse",
            "val_rmse",
            "train_r2",
            "val_r2",
            "train_abs_loss",
            "train_emi_loss",
            "train_plqy_loss",
            "train_em_loss",
            "val_abs_loss",
            "val_emi_loss",
            "val_plqy_loss",
            "val_em_loss",
            "train_abs_mae",
            "train_emi_mae",
            "train_plqy_mae",
            "train_em_mae",
            "val_abs_mae",
            "val_emi_mae",
            "val_plqy_mae",
            "val_em_mae",
            "train_abs_r2",
            "train_emi_r2",
            "train_plqy_r2",
            "train_em_r2",
            "val_abs_r2",
            "val_emi_r2",
            "val_plqy_r2",
            "val_em_r2",
            "train_total_rows_before",
            "train_kept_rows",
            "train_dropped_total_rows",
            "train_solute_skipped_rows",
            "train_solute_invalid_count",
            "train_solute_missing_count",
            "train_solvent_smiles_required",
            "train_solvent_skipped_rows",
            "train_solvent_invalid_count",
            "train_solvent_missing_count",
            "val_total_rows_before",
            "val_kept_rows",
            "val_dropped_total_rows",
            "val_solute_skipped_rows",
            "val_solute_invalid_count",
            "val_solute_missing_count",
            "val_solvent_smiles_required",
            "val_solvent_skipped_rows",
            "val_solvent_invalid_count",
            "val_solvent_missing_count",
            "test_total_rows_before",
            "test_kept_rows",
            "test_dropped_total_rows",
            "test_solute_skipped_rows",
            "test_solute_invalid_count",
            "test_solute_missing_count",
            "test_solvent_smiles_required",
            "test_solvent_skipped_rows",
            "test_solvent_invalid_count",
            "test_solvent_missing_count",
        ]
        writer.writerow(header)

        n_epochs = len(train_loss_arr)
        for i in range(n_epochs):
            is_last = (i == n_epochs - 1)
            writer.writerow([
                i + 1,
                1 if (i + 1) == best_epoch else 0,
                train_loss_arr[i],
                val_loss_arr[i],
                train_mae_arr[i],
                val_mae_arr[i],
                train_rmse_arr[i],
                val_rmse_arr[i],
                train_r2_arr[i],
                val_r2_arr[i],
                train_task_loss_arr[i, 0],
                train_task_loss_arr[i, 1],
                train_task_loss_arr[i, 2],
                train_task_loss_arr[i, 3],
                val_task_loss_arr[i, 0],
                val_task_loss_arr[i, 1],
                val_task_loss_arr[i, 2],
                val_task_loss_arr[i, 3],
                train_task_mae_arr[i, 0],
                train_task_mae_arr[i, 1],
                train_task_mae_arr[i, 2],
                train_task_mae_arr[i, 3],
                val_task_mae_arr[i, 0],
                val_task_mae_arr[i, 1],
                val_task_mae_arr[i, 2],
                val_task_mae_arr[i, 3],
                train_task_r2_arr[i, 0],
                train_task_r2_arr[i, 1],
                train_task_r2_arr[i, 2],
                train_task_r2_arr[i, 3],
                val_task_r2_arr[i, 0],
                val_task_r2_arr[i, 1],
                val_task_r2_arr[i, 2],
                val_task_r2_arr[i, 3],
                train_total_rows_before_arr[0] if is_last else "",
                train_kept_rows_arr[0] if is_last else "",
                train_dropped_total_rows_arr[0] if is_last else "",
                train_solute_skipped_rows_arr[0] if is_last else "",
                train_solute_invalid_count_arr[0] if is_last else "",
                train_solute_missing_count_arr[0] if is_last else "",
                train_solvent_smiles_required_arr[0] if is_last else "",
                train_solvent_skipped_rows_arr[0] if is_last else "",
                train_solvent_invalid_count_arr[0] if is_last else "",
                train_solvent_missing_count_arr[0] if is_last else "",
                val_total_rows_before_arr[0] if is_last else "",
                val_kept_rows_arr[0] if is_last else "",
                val_dropped_total_rows_arr[0] if is_last else "",
                val_solute_skipped_rows_arr[0] if is_last else "",
                val_solute_invalid_count_arr[0] if is_last else "",
                val_solute_missing_count_arr[0] if is_last else "",
                val_solvent_smiles_required_arr[0] if is_last else "",
                val_solvent_skipped_rows_arr[0] if is_last else "",
                val_solvent_invalid_count_arr[0] if is_last else "",
                val_solvent_missing_count_arr[0] if is_last else "",
                test_total_rows_before_arr[0] if is_last else "",
                test_kept_rows_arr[0] if is_last else "",
                test_dropped_total_rows_arr[0] if is_last else "",
                test_solute_skipped_rows_arr[0] if is_last else "",
                test_solute_invalid_count_arr[0] if is_last else "",
                test_solute_missing_count_arr[0] if is_last else "",
                test_solvent_smiles_required_arr[0] if is_last else "",
                test_solvent_skipped_rows_arr[0] if is_last else "",
                test_solvent_invalid_count_arr[0] if is_last else "",
                test_solvent_missing_count_arr[0] if is_last else "",
            ])

    print(f"[CSV] saved -> {log_csv_path}")
