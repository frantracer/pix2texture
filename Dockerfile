FROM frantracer/tensorflow-2-gpu:cuda-10.1

WORKDIR /tf

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY notebooks/checkpoints ./notebooks/checkpoints/
COPY notebooks/*ipynb ./notebooks/
COPY *py ./

CMD /usr/bin/python3 /usr/local/bin/jupyter-notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root