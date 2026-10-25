PY ?= python3

data:            ## Télécharge SROIE et lance l'OCR Tesseract
	mkdir -p data && test -d data/sroie || (git clone --depth 1 https://github.com/zzzDavid/ICDAR-2019-SROIE.git /tmp/sroie && ln -s /tmp/sroie/data data/sroie)
	$(PY) scripts/prepare_data.py

extract:         ## Entraîne et évalue l'extracteur
	$(PY) scripts/train_extractor.py

fraud:           ## Entraîne et évalue le score de risque
	$(PY) scripts/train_fraud.py
	$(PY) scripts/eval_image_tampering.py

figures:
	$(PY) scripts/make_figures.py

all: data extract fraud figures

test:
	$(PY) -m unittest discover tests

demo:
	$(PY) app/app.py
