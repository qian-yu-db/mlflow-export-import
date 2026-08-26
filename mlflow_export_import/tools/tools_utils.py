from mlflow_export_import.common import utils
from mlflow_export_import.common.iterators import SearchRegisteredModelsIterator, SearchModelVersionsIterator
from mlflow_export_import.client import client_utils

def search_model_versions(client, filter):
    if utils.calling_databricks(client_utils.create_dbx_client(client)):
        models = list(SearchRegisteredModelsIterator(client, filter=filter))
        versions = []
        for model in models:
            try:
                _versions = list(SearchModelVersionsIterator(client, filter=f"name='{model.name}'"))
                versions += _versions
            except Exception as e:
                print(f"ERROR: registered model '{model.name}': {e}")
        return versions
    else:
        return list(SearchModelVersionsIterator(client, filter=filter))
