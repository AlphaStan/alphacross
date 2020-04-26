run-tests:
	python -m pytest

train-model:
	cd src/models/ && python train_model.py
