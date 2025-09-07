.PHONY: train serve test

train:
	python -m src.train

serve:
	MODEL_PATH=models/model.pkl python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q
