LOST_DIR=$(PWD)/install
LOST_DATA_DIR=$(LOST_DIR)/data/data
LOST_PIPELINE_DIR=$(LOST_DATA_DIR)/pipes

update:
	rm -rf  $(LOST_PIPELINE_DIR)/mvp || echo 0
	cp -r mvp/pipeline $(LOST_PIPELINE_DIR)/mvp
	docker exec -it lost /opt/conda/envs/lost/bin/python update_pipe_project.py /home/lost/data/pipes/mvp

build:
	docker build -t lost-custom:0.01 -f docker/executors/lost-cv-custom/Dockerfile .

down:
	cd $(LOST_DIR)/docker ; docker-compose down

up: down build
	cd $(LOST_DIR)/docker ; docker-compose up

uninstall:
	$(MAKE) down || echo 0
	rm -rf $(LOST_DIR)

install: uninstall
	cd docker/quick_setup/ ; python quick_setup.py $(LOST_DIR)
	mkdir -p $(LOST_DATA_DIR)/media
	cp -r mvp/20x $(LOST_DATA_DIR)/media/
	$(MAKE) up