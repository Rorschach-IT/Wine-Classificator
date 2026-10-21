# Design decisions and limitations

[Documentation index](README.md)

These are retrospective descriptions of the supplied design, not a claim that formal decision records existed before implementation.

## Decision 1: small function-oriented Python modules

Context: an educational experiment needs clarity without a web framework or complex packaging.

Decision: separate loading, exploration, visualization, configuration, evaluation and orchestration.

Consequences: functions are easy to read and modify; script-style imports mean package invocation and distribution need deliberate refactoring if added later. No class hierarchy is required by the present workflow.

## Decision 2: pipelines for fold-local preprocessing

Context: scaling before splitting or before CV can leak validation information.

Decision: scaler and estimator are evaluated as one Pipeline for logistic regression and k-NN.

Consequences: fold-local fitting is automatic. New learned transformations, including imputation or feature selection, should be placed inside the pipeline rather than performed globally.

## Decision 3: fixed holdout plus stratified CV

Context: 178 samples provide limited evaluation data.

Decision: reserve 20 percent for final testing and use five-fold stratified CV within the remainder.

Consequences: a clear final evaluation boundary, but a noisy 36-sample test. Nested CV or repeated evaluation would require additional methodology and compute; neither is implemented.

## Decision 4: macro F1 chooses configurations and family

Context: the assignment requires several metrics and three classes.

Decision: report all required metrics but establish macro F1 as the selection criterion.

Consequences: equal class weighting and a clear objective. Macro F1 is not necessarily the correct objective for every real-world cost model. Exact ties currently follow dictionary order instead of an explicit simplicity or latency preference.

## Decision 5: file-based local outputs

Context: results should be easy to inspect and include in a report.

Decision: CSV, PNG and TXT in a fixed results directory.

Consequences: portable inspection, but reruns overwrite outputs; no transaction, provenance manifest, persisted fitted model or unique run identifier.

## Current limitations

- Educational benchmark, not validated production wine recognition.
- No external validation dataset or domain-shift assessment.
- No confidence intervals or formal significance tests.
- No probability calibration, probability exports or cost-sensitive objective.
- No implemented majority-class baseline, nested CV or repeated CV.
- No automatic schema validation or missing-value treatment.
- No saved model or separate inference entry point.
- No automated test suite or CI in the supplied code.
- No run-level metadata capture or dependency policy beyond suggested freezing.
- Holdout error inspection is diagnostic, not permission to keep tuning against the same test.

## Sensible next improvements

1. Add tests for data contracts and output schemas.
2. Record run metadata and isolate result directories.
3. Add type annotations, docstrings and explicit input validation.
4. Add a DummyClassifier baseline for context.
5. Consider nested or repeated validation if the goal becomes robust model comparison rather than a first coursework experiment.
6. Add model persistence and an inference contract only if reuse on new measurements is required.

These are proposals. Update source code, tests and documentation together before describing them as implemented.
