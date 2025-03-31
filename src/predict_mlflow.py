import mlflow

model_name = "fasttext4"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

# list_libs = ["vendeur d'huitres", "boulanger"]
list_libs = ["coiffeur, & 98789", "COIFFEUR"]

results = model.predict(list_libs, params={"k": 1})
print(results)

description = "boulangerie"
nb_echoes_max = 1

query = {
        "query": [description],
        "k": nb_echoes_max,
        }

results = model.predict(query["query"], params={"k": 1})
print(results)

