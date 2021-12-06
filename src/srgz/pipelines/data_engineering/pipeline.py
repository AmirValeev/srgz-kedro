from kedro.pipeline import Pipeline, node
from .nodes import data_preparation, feature_prep

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=feature_prep,
                inputs='features_raw',
                outputs='features',
                name='features'
            ),
            node(
                func=data_preparation,
                inputs=['features', 'df1', 'df2'],
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name='data_preparation'
            )

        ]
    )