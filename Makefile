install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src tests

test:
	pytest tests

train:
	python src/train.py