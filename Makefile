build:
	USE_CYTHON=1 python3 setup.py build_ext --inplace

test:
	python3 -m pytest tests/ -q --tb=short -x

wheel:
	USE_CYTHON=1 python3 -m build --wheel

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f governance_core/*.so governance_core/*.c

harden:
	strip -x governance_core/*.so
