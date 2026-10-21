# How-to guides

[Documentation index](README.md)

## Change a hyperparameter search

1. Edit the appropriate `params` dictionary in `main/models.py`.
2. Keep the `model__` prefix because the estimator step is named `model`.
3. Check that the parameter is supported by that estimator.
4. Keep the test set out of the decision.
5. Run the experiment again and document the changed search space.

Expanding a grid increases work multiplicatively. More candidates also increase the opportunity to select a configuration that benefits from CV noise. Preserve the prior run before rerunning.

## Change the random seed

Edit `RANDOM_STATE` in `main/dataset.py`. Modules import it for split and validation behavior. Changing it alters the experiment, not merely formatting. Record it with results and do not search seeds until a favorable test score appears.

## Reduce CPU usage

Replace `n_jobs=-1` in the three evaluation calls (`cross_validate`, `GridSearchCV`, `learning_curve`) with `1` or another appropriate worker limit. This is a source-code change; no environment-independent project setting currently controls all three.

## Run only exploratory analysis

Use an isolated script or temporary development branch that calls `load_dataset`, `split_dataset`, `analyze_dataset` and `create_visualizations` without model evaluation. Ensure the output Path exists. Avoid leaving a partially disabled production entry point committed without documenting the change.

## Translate runtime labels

Edit user-facing print strings and Matplotlib title/axis strings in `main.py`, `analysis.py`, `visualization.py` and `evaluation.py`. Do not rename feature columns or CSV field names merely to translate presentation. Current documentation is English, but the tutorial source uses Polish console and plot labels.

## Handle missing features in an extended dataset

This is a proposed extension, not current behavior. Decide whether missing values should be rejected or imputed. If imputing, put the imputer inside the estimator Pipeline before scaling so it is fitted only on fold-training samples. Add tests for the chosen behavior. A printed missing count is not input validation.

## Add another classifier

Add a new entry with `estimator` and `params` to `get_model_configs()`. A Pipeline with its classifier step named `model` keeps parameter naming consistent. Check prediction labels, metrics, preprocessing and runtime. A new entry automatically participates in comparison and selection; update expected row counts, output inventory and methodological documentation.

## Preserve an experiment

Before a new run, copy the completed results to an explicitly named archive outside the active output directory. Record source commit, interpreter version, dependency versions, seed and settings. Do not label a directory with a winning score before the run has completed. A timestamped run directory and metadata manifest would be useful future enhancements, but are not implemented.

## Add a repository-root documentation link

Add this link to the existing root README without replacing its contents:

```markdown
## Documentation

See the [project documentation](docs/README.md) for setup, architecture,
API reference, experiment methodology and troubleshooting.
```
