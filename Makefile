build:
	USE_CYTHON=1 python3 setup.py build_ext --inplace

test:
	python3 -m pytest tests/ -q --tb=short -x

wheel: build
	@# Remove source files (except __init__.py) so wheel only contains .so + __init__.py
	@for f in governance_core/*.py; do \
		[ "$$(basename $$f)" = "__init__.py" ] || mv "$$f" "$$f.bak"; \
	done
	@rm -f governance_core/*.c
	USE_CYTHON=0 python3 -m build --wheel
	@# Restore source files
	@for f in governance_core/*.py.bak; do \
		mv "$$f" "$${f%.bak}"; \
	done

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f governance_core/*.so governance_core/*.c

harden:
	strip -x governance_core/*.so
