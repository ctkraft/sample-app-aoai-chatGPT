 #!/bin/sh

echo 'Logging into Azure tenant'
az login --tenant 031d8bee-cd40-4791-b63c-bc890be44c22

echo 'Running data_preparation.py'
python data_preparation.py --njobs=1 --embedding-model-endpoint=https://aoai-fema-ocfo-gpt2.openai.azure.com/openai/deployments/dep-embeddings/embeddings?api-version=2023-03-15-preview


