# Cloud_Progamm

# CS 643 Cloud Computing Programming assignment 2
### Vedant Kadu vk63

### Goal: 

The purpose of this individual assignment is to learn how to develop parallel machine learning (ML) applications in Amazon AWS cloud platform. Specifically, you will learn: (1) how to use Apache Spark to  train an ML model in parallel on multiple EC2 instances; (2) how to use Spark’s MLlib to develop and use  an ML model in the cloud; (3) How to use  Docker to create a container for your ML model to simplify model deployment.  

### Description:

Using Flintrock I created Spark Cluster for my application. I used the Python as my programming language. The input of my model is Training Dataset and Validation dataset. I used WinSCP to give Dataset input for all m slave nodes and only master node contains Python File.
Login Into AWS Account
Initially created single instance and downloaded New Key Pair. Converted Key Pair using PuTTy gen and Started created instance using PuTTy using Public IPv4 DNS. 

### Credentials:
Using PuTTy Set the credentials for our instances. Using nano credentials command. Set the Access key, Secret Key along with the aws session token

### Flintrock 
Installed the Flintrock on the instance using pip3 install flintrock command. After installing I configured the ymal file with the instance credentials along with that I specified the number of nodes that I wanted to create.

### inbound Rules
After Creating the Cluster add the SSH inbound rule to the master node for port 22

### Login into Master node
Using command Flintrock start <ClusterName>. I started the cluster and Using the command Flintrock login <Clustername>

### WinSCP
Start WinSCP and login into master node and all the slave nodes using the Public IPv4 DNS. I upload the Python code file in master node along with the two dataset file. and slave also contains the datafile.

### Run Code
 Run the code using command:
/home/ec2-user/spark/bin/spark-submit --master spark://<Ip address of master node>ec2.internal:7077 /home/ec2-user/Untitled.py

