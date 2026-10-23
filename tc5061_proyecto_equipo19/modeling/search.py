"""Búsqueda reproducible: un run padre y tres runs hijos (P1 Ridge)."""
import json

import mlflow

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.modeling.artifacts import log_run_config
from tc5061_proyecto_equipo19.modeling.mlflow_utils import (
    _new_mlflow_run, configurar_experimento, contexto_comun,
)
from tc5061_proyecto_equipo19.modeling.train_P1 import preparar_datos, train_model


def search_p1_ridge():
    configurar_experimento()
    alphas = cfg.P1_RIDGE_SEARCH_ALPHAS
    # Cargar y preparar 24 meses una sola vez para todas las combinaciones.
    prepared = preparar_datos()
    params, tags = contexto_comun()
    with _new_mlflow_run("P1-search-ridge") as parent:
        mlflow.set_tags({**tags, "question": "p1", "model_family": "regression",
                         "stage_of_work": "search", "pregunta": "P1"})
        mlflow.log_params({**params, "search_method": "grid",
                           "search_space_alpha": json.dumps(alphas)})
        mlflow.log_artifact(str(cfg.MANIFEST_PATH), artifact_path="data")
        log_run_config({"question": "p1", "model": "ridge", "alphas": alphas,
                        "selection_metric": "valid_mae"})
        rows = []
        for alpha in alphas:
            rows.append(train_model(model="ridge", params={"alpha": alpha},
                                    nested=True, stage_of_work="search",
                                    prepared_splits=prepared))
        best = min(rows, key=lambda row: row["valid_mae"])
        mlflow.log_metric("best_valid_mae", best["valid_mae"])
        mlflow.set_tag("best_child_run_id", best["run_id"])
        print("Corrida padre:", parent.info.run_id)
        print("Ganador provisional (validación):", best)
        return {"parent_run_id": parent.info.run_id, "children": rows}


if __name__ == "__main__":
    search_p1_ridge()