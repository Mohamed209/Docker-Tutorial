# Docker-Flask Tutorial #
 repo conducts example of how to use docker to deploy machine learning model
 
1- in model folder (/web/model) simple linear regression model is trained to predict employee salary based on years of experience

2- model saves it weights in /web directory

3- model is deployed on localhost using flask script in app.py
 
## Dockerize you model ##
1- first install docker (https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)

2- in project directory create dockerfile (check tutorial https://odewahn.github.io/docker-jumpstart/building-images-with-dockerfiles.html)

3- build the docker image using command "$ docker build -t flask-sample:latest ."

4- run the container to check results "$ docker run -p 5000:5000 flask-sample" , now you have dockerized your app well done ! :)

6- you can also run the container using docker decompse first stop running container "docker stop [YOUR CONTAINER NUMBER HERE]" to find container number run "$ docker ps -a"

7- to run using decompose run "$ docker-compose up"


## References ##
https://codefresh.io/docker-tutorial/hello-whale-getting-started-docker-flask/
