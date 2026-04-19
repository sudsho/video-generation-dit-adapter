.PHONY: install test lint train distill-anime distill-cinematic api streamlit docker docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	python -m compileall -q src tests

train:
	bash scripts/train_motion.sh configs/base.yaml runs/motion

distill-anime:
	bash scripts/distill_lora.sh configs/style_anime.yaml runs/motion/motion_final.pt runs/lora/anime

distill-cinematic:
	bash scripts/distill_lora.sh configs/style_cinematic.yaml runs/motion/motion_final.pt runs/lora/cinematic

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

streamlit:
	streamlit run streamlit_app.py

docker:
	docker build -t video-motion-adapter:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf out/*.mp4 __pycache__ .pytest_cache
