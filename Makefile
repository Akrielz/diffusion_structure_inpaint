.PHONY: build
build:
	export DOCKER_BUILDKIT=1
	docker build -t foldingdiff .

.PHONY: run
run:
	docker run --rm -it foldingdiff