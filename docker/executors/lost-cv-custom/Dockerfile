ARG base_image=l3pcv/lost-base:latest
FROM $base_image
# Add Code 
ADD /docker/executors/lost-cv-custom/entrypoint.sh /
ADD /backend/ /code/backend

# notice these are in the OTHER folder that I copied from
ADD /docker/executors/lost-cv-custom/environment.yml .
ADD /docker/executors/lost-cv-custom/requirements.txt .

# RUN /bin/bash -c "source /opt/conda/bin/activate && conda create -n lost-cv pytorch torchvision cpuonly -c pytorch"
RUN /bin/bash -c "source /opt/conda/bin/activate && conda env create -f environment.yml && conda clean -ay"
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && pip install --no-cache-dir -r requirements.txt"
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && conda install pyyaml"
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html"
# tensorboard
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && pip install --upgrade pip"
RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && pip install tensorflow==2.3.0"
# RUN /bin/bash -c "source /opt/conda/bin/activate lost-cv && conda install 'tensorboard>=1.5'" 