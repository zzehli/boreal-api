endpoint = "https://tropiquesloop-8903-ai925517699996.openai.azure.com/"
model_name = "text-embedding-3-large"
deployment = "text-embedding-3-large"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    endpoint=endpoint,
    credential=AzureKeyCredential("<API_KEY>")
)