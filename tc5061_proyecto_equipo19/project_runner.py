"""Ruteo de los entry points de MLflow Projects; defaults solo de config.py."""
import argparse

from tc5061_proyecto_equipo19 import config as cfg


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("ingest")
    search = sub.add_parser("search")
    search.add_argument("--question", choices=["p1"], default="p1")
    train = sub.add_parser("train")
    train.add_argument("--question", choices=["p1", "p2", "p3"], required=True)
    train.add_argument("--model", default="default")
    train.add_argument("--alpha", type=float, default=-1.0)
    train.add_argument("--C", type=float, default=-1.0)
    train.add_argument("--learning-rate", type=float, default=-1.0)
    train.add_argument("--n-estimators", type=int, default=-1)
    train.add_argument("--max-depth", type=int, default=-1)
    train.add_argument("--max-leaf-nodes", type=int, default=-1)
    train.add_argument("--min-samples-leaf", type=int, default=-1)
    train.add_argument("--max-train-rows", type=int, default=-1)
    train.add_argument("--n-clusters", type=int, default=-1)
    train.add_argument("--n-init", type=int, default=-1)
    train.add_argument("--max-iter", type=int, default=-1)
    train.add_argument("--random-state", type=int, default=-1)
    train.add_argument("--model-type", choices=["baseline", "candidate", "perfil", "hibrido"], default="candidate")
    train.add_argument("--n-components", type=int, default=1)
    train.add_argument("--volume-weight", type=float, default=1.0)
    return parser.parse_args()


def _default(value, default):
    return default if value == -1 else value


def main():
    args = parse_args()
    if args.action == "ingest":
        from tc5061_proyecto_equipo19.dataset import run_ingestion
        run_ingestion()
        return
    if args.action == "search":
        from tc5061_proyecto_equipo19.modeling.search import search_p1_ridge
        search_p1_ridge()
        return
    model = args.model
    if args.question == "p1":
        from tc5061_proyecto_equipo19.modeling.train_P1 import train_model
        model = cfg.P1_DEFAULT_MODEL if model == "default" else model
        train_model(model=model, params={
            "alpha": _default(args.alpha, cfg.P1_RIDGE_ALPHA),
            "n_estimators": _default(args.n_estimators, cfg.P1_RF_N_ESTIMATORS),
            "max_depth": _default(args.max_depth, cfg.P1_RF_MAX_DEPTH),
        })
    elif args.question == "p2":
        from tc5061_proyecto_equipo19.modeling.train_p2 import train_model
        model = cfg.P2_DEFAULT_MODEL if model == "default" else model
        opts = {"C": _default(args.C, cfg.P2_LOGREG_C),
                "max_iter": _default(args.max_iter, cfg.P2_HGB_MAX_ITER),
                "learning_rate": _default(args.learning_rate, cfg.P2_HGB_LEARNING_RATE),
                "max_leaf_nodes": _default(args.max_leaf_nodes, cfg.P2_HGB_MAX_LEAF_NODES),
                "min_samples_leaf": _default(args.min_samples_leaf, cfg.P2_HGB_MIN_SAMPLES_LEAF)}
        if model == "logreg":
            # max_iter diferente de HGB si no se especificó explícitamente.
            opts["max_iter"] = _default(args.max_iter, cfg.P2_LOGREG_MAX_ITER)
        train_model(model=model, params=opts,
                    max_train_rows=_default(args.max_train_rows, cfg.MAX_FILAS_ENTRENAMIENTO))
    else:
        from tc5061_proyecto_equipo19.modeling.train_P3 import train_model
        model = cfg.P3_DEFAULT_MODEL if model == "default" else model
        train_model(model=model, params={
            "n_clusters": _default(args.n_clusters, cfg.P3_N_CLUSTERS),
            "n_init": _default(args.n_init, cfg.P3_N_INIT),
            "max_iter": _default(args.max_iter, cfg.P3_MAX_ITER),
            "random_state": _default(args.random_state, cfg.RANDOM_SEED),
            "model_type": args.model_type,
            "n_components": _default(args.n_components, cfg.P3_N_COMPONENTS),
            "volume_weight": _default(args.volume_weight, cfg.P3_VOLUME_WEIGHT),
        })


if __name__ == "__main__":
    main()