from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class EarlyStoppingState:
    """
    。
    """
    best_score: Optional[float] = None
    best_epoch: int = 0
    bad_epochs: int = 0
    should_stop: bool = False
    last_improved: bool = False


class EarlyStopping:
    """
    。

    ：
    -  val_r2（mode='max'）
    -  val_loss（mode='min'）

    
    ----
    patience : int
         epoch “”。
    min_delta : float
        “”。
         mode='max' ， > best_score + min_delta 。
    mode : str
        'max'  'min'
        - 'max': ， R2
        - 'min': ， loss
    save_path : str | Path | None
        ， checkpoint。
    verbose : bool
        。
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "max",
        save_path: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience  >= 1， {patience}")
        if mode not in {"max", "min"}:
            raise ValueError(f"mode  'max'  'min'， {mode}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.save_path = str(save_path) if save_path is not None else None
        self.verbose = bool(verbose)

        self.state = EarlyStoppingState()

    def reset(self) -> None:
        """。"""
        self.state = EarlyStoppingState()

    @property
    def best_score(self) -> Optional[float]:
        return self.state.best_score

    @property
    def best_epoch(self) -> int:
        return self.state.best_epoch

    @property
    def bad_epochs(self) -> int:
        return self.state.bad_epochs

    @property
    def should_stop(self) -> bool:
        return self.state.should_stop

    @property
    def last_improved(self) -> bool:
        return self.state.last_improved

    def _is_improvement(self, score: float) -> bool:
        if self.state.best_score is None:
            return True

        if self.mode == "max":
            return score > self.state.best_score + self.min_delta
        return score < self.state.best_score - self.min_delta

    def _save_checkpoint(
        self,
        score: float,
        epoch: int,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.save_path is None:
            return

        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "monitor_score": float(score),
            "early_stopping": self.state_dict(),
        }

        if model is not None:
            payload["model_state_dict"] = model.state_dict()

        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state is not None:
            payload["extra_state"] = extra_state

        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def step(
        self,
        score: float,
        epoch: int,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
         epoch ，。

        
        ----
        score : float
            ， val_r2  val_loss。
        epoch : int
             epoch 。
        model : nn.Module | None
             save_path ， model.state_dict()。
        optimizer : torch.optim.Optimizer | None
             save_path ，。
        extra_state : dict | None
             checkpoint 。

        
        ----
        bool
            True ；False 。
        """
        score = float(score)
        improved = self._is_improvement(score)
        self.state.last_improved = improved

        if improved:
            self.state.best_score = score
            self.state.best_epoch = int(epoch)
            self.state.bad_epochs = 0
            self.state.should_stop = False
            self._save_checkpoint(
                score=score,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                extra_state=extra_state,
            )
            if self.verbose:
                print(
                    f"[EarlyStopping] improved: best_score={self.state.best_score:.6f} "
                    f"at epoch={self.state.best_epoch}"
                )
            return False

        self.state.bad_epochs += 1
        if self.verbose:
            print(
                f"[EarlyStopping] no improvement: {self.state.bad_epochs}/{self.patience} "
                f"(best_score={self.state.best_score})"
            )

        if self.state.bad_epochs >= self.patience:
            self.state.should_stop = True
            if self.verbose:
                print(
                    f"[EarlyStopping] stop triggered at epoch={epoch}. "
                    f"Best epoch={self.state.best_epoch}, best_score={self.state.best_score}"
                )
            return True

        self.state.should_stop = False
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "save_path": self.save_path,
            "verbose": self.verbose,
            "state": asdict(self.state),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.patience = int(state_dict["patience"])
        self.min_delta = float(state_dict["min_delta"])
        self.mode = str(state_dict["mode"])
        self.save_path = state_dict.get("save_path", None)
        self.verbose = bool(state_dict.get("verbose", True))
        self.state = EarlyStoppingState(**state_dict["state"])

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(mode={self.mode!r}, patience={self.patience}, "
            f"min_delta={self.min_delta}, best_score={self.best_score}, "
            f"best_epoch={self.best_epoch}, bad_epochs={self.bad_epochs}, "
            f"should_stop={self.should_stop})"
        )


def load_early_stopping_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
     EarlyStopping  checkpoint。
    """
    return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    
    es = EarlyStopping(
        patience=5,
        min_delta=1e-4,
        mode="max",
        save_path="best_model.pt",
        verbose=True,
    )

    fake_scores = [0.30, 0.35, 0.351, 0.349, 0.348, 0.3485, 0.3487, 0.3486]
    for ep, score in enumerate(fake_scores, start=1):
        stop = es.step(score=score, epoch=ep)
        if stop:
            break

    print(es)
