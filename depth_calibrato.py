"""
depth_calibrato.py
------------------
Modulo per applicare la calibrazione SCDepthV3 → metri sul Raspberry Pi 5.
Carica i parametri da calib_params.json generato da calibra_depth.py.

Uso:
    from depth_calibrato import DepthCalibrator
    cal = DepthCalibrator("calib_params.json")
    depth_metri = cal.to_meters(raw_depth_from_hailo)
"""

import json
import numpy as np


class DepthCalibrator:
    """Converte depth relativa SCDepthV3 → depth metrica usando calibrazione."""

    SUPPORTED_TYPES = {"affine", "exp", "exp_affine", "neg_exp"}

    def __init__(self, calib_json_path: str):
        with open(calib_json_path) as f:
            data = json.load(f)

        # struttura attesa: { "best": {...}, "global_fit": {...}, ... }
        if "best" in data:
            self.params = data["best"]
        else:
            # fallback: file direttamente con i parametri
            self.params = data

        self.type = self.params["type"]
        if self.type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Tipo calibrazione non supportato: {self.type}. "
                f"Supportati: {self.SUPPORTED_TYPES}"
            )

        self.a = float(self.params["a"])
        self.b = float(self.params["b"])
        self.c = float(self.params.get("c", 0.0))
        self.metrics = self.params.get("metrics", {})
        self.fit_range = self.params.get("fit_range_m", [0.3, 50.0])

        formula = self._formula_str()
        print(f"[DepthCalibrator] {self.type}: {formula}")
        if self.metrics:
            mae = self.metrics.get("mae_m", "?")
            pct = self.metrics.get("pct_within_1m", "?")
            print(f"  MAE={mae} m  |  pixel <1m: {pct}%  |  range fit: {self.fit_range} m")

    def _formula_str(self) -> str:
        if self.type == "affine":
            return f"d = {self.a:.4f} * sc + {self.b:.4f}"
        if self.type == "exp":
            return f"d = {self.a:.4f} * exp({self.b:.4f} * sc) + {self.c:.4f}"
        if self.type == "exp_affine":
            return f"d = {self.a:.4f} * exp(sc) + {self.b:.4f}"
        if self.type == "neg_exp":
            return f"d = {self.a:.4f} * exp(-sc) + {self.b:.4f}"
        return "?"

    def to_meters(
        self,
        sc_depth: np.ndarray,
        clamp_min: float = 0.0,
        clamp_max: float = 300.0,
    ) -> np.ndarray:
        """
        Converte output raw SCDepthV3 in depth metrica (metri).

        Args:
            sc_depth: depth map dal modello (qualsiasi shape)
            clamp_min/max: limiti finali in metri

        Returns:
            stessa shape, valori in metri (float32)
        """
        sc = sc_depth.astype(np.float32)

        if self.type == "affine":
            depth_m = self.a * sc + self.b
        elif self.type == "exp":
            depth_m = self.a * np.exp(self.b * sc) + self.c
        elif self.type == "exp_affine":
            depth_m = self.a * np.exp(sc) + self.b
        elif self.type == "neg_exp":
            depth_m = self.a * np.exp(-sc) + self.b
        else:
            raise ValueError(f"Tipo non supportato: {self.type}")

        return np.clip(depth_m, clamp_min, clamp_max).astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cal = DepthCalibrator(sys.argv[1])
        test_sc = np.array([[-6.5, -5.0, -3.5]], dtype=np.float32)
        out = cal.to_meters(test_sc)
        print(f"\nTest input sc:   {test_sc[0]}")
        print(f"Output metri:    {out[0]}")
