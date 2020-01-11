run-tests:
	cd src/test/ && python -m pytest

train-model:
	cd src/models/ && python train_model.py