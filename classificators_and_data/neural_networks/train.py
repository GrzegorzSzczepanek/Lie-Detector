from neural_networks.constants import (
    IDS_TO_RUN_NAMES_FILE_PATH,
    IDS_TO_RUN_NAMES_SEPARATOR,
    MODEL_CONFIG_PATH,
    MODELS_PATH,
)
from neural_networks.models.config_to_run import CONFIG_TO_RUN
from neural_networks.trainer.single_param_search import SingleParamSearch


def main() -> None:
    run_name = CONFIG_TO_RUN["run_name"]
    if _is_run_name_duplicate(run_name):
        print(f"Run name '{run_name}' already used. Be more creative.")
        return

    searcher = SingleParamSearch(CONFIG_TO_RUN)
    searcher.search()

    run_id = searcher.get_run_id()
    _add_run_id_and_name(run_id, run_name)
    _copy_run_config(run_id, CONFIG_TO_RUN["model_name"])


def _is_run_name_duplicate(run_name: str) -> bool:
    if not IDS_TO_RUN_NAMES_FILE_PATH.exists():
        return False

    with open(IDS_TO_RUN_NAMES_FILE_PATH, encoding="utf-8") as f:
        ids_to_run_names = f.read().splitlines()

    run_names = set(
        line.split(IDS_TO_RUN_NAMES_SEPARATOR)[1] for line in ids_to_run_names
    )

    return run_name in run_names


def _add_run_id_and_name(run_id: str, run_name: str) -> None:
    with open(IDS_TO_RUN_NAMES_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"{run_id}{IDS_TO_RUN_NAMES_SEPARATOR}{run_name}\n")


def _copy_run_config(run_id: str, model_name: str) -> None:
    with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
        model_config_text = f.read()

    model_dir = MODELS_PATH / model_name
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / f"{run_id}.py", "w", encoding="utf-8") as f:
        f.write(model_config_text)


if __name__ == "__main__":
    main()
