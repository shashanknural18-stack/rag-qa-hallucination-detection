.PHONY: setup test test-all test-slow app eval eval-fast compile

setup:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-all:
	pytest tests/ -v -m ""

test-slow:
	pytest tests/test_end_to_end.py -v -m slow

app:
	streamlit run app.py

eval:
	python run_eval.py

eval-fast:
	python run_eval.py --limit 5 --no-judge

compile:
	python -m compileall src tests app.py run_eval.py demo_retrieval.py demo_generation.py demo_detection.py

