docker build -t tfgen .
docker run -it --rm --runtime=nvidia tfgen /bin/bash
