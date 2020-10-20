NAME=dermclass-api
COMMIT_ID=$(shell git rev-parse HEAD)

build-dermclass-api-heroku:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t registry.heroku.com/$(NAME)/web:$(COMMIT_ID) .

push-dermclass-api-heroku:
	docker push registry.heroku.com/${HEROKU_APP_NAME}/web:$(COMMIT_ID)
