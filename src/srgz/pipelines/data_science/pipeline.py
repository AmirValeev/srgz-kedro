from kedro.pipeline import Pipeline, node

from .nodes import model_fitting, params_optimization, model_testing


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=params_optimization,
                inputs=["X_train", "y_train"],
                outputs="lgb_opt",
                name="params_optimization",
            ),
            node(
                func=model_fitting,
                inputs=['lgb_opt', "X_train", "y_train", "X_test", "y_test"],
                outputs="lgb_model",
                name="model_fitting",
            ),
            node(
                func=model_testing,
                inputs=['X_test', 'y_test', 'lgb_model'],
                outputs="score",
                name='model_scoring'
                )
        ]
    )