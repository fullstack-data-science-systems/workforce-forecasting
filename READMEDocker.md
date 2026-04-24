# Docker, AWS ECR, ECS & EC2 Deployment Guide

This guide covers: building the Docker image, pushing to AWS ECR, and deploying to both ECS Fargate and EC2 Ubuntu.

---

## Prerequisites

| Tool | Version | Install Link |
|------|---------|--------------|
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop |
| AWS CLI | v2 | https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html |
| AWS Account | — | https://aws.amazon.com |

---

## Part 1: Build the Docker Image

### Step 1.1 — Ensure model files are present

Confirm these files exist in your project directory before building:

```
employment_forecast_lstm_final.h5
employment_forecast_gru_final.h5
employment_forecast_cnn_final.h5
scaler.pkl           (optional – generated at container startup if missing)
example_data.csv
```

### Step 1.2 — Build the image

```bash
# Windows / Mac / Linux (from project root):
docker build -t employment-forecasting:latest .

# Verify the build succeeded
docker images | grep employment-forecasting
```

### Step 1.3 — Test locally

```bash
docker run -p 8000:8000 employment-forecasting:latest
```

Open http://localhost:8000/health — you should see `"status": "healthy"`.

### Step 1.4 — Stop the container

```bash
# Find container ID
docker ps

# Stop
docker stop <container_id>
```

---

## Part 2: Push to AWS ECR

### Step 2.1 — Configure AWS CLI

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format: json
```

Verify:
```bash
aws sts get-caller-identity
```

### Step 2.2 — Create ECR repository

```bash
aws ecr create-repository \
  --repository-name employment-forecasting \
  --region us-east-1
```

Note the `repositoryUri` from the output. It looks like:
`123456789012.dkr.ecr.us-east-1.amazonaws.com/employment-forecasting`

### Step 2.3 — Authenticate Docker to ECR

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com
```

Expected: `Login Succeeded`

### Step 2.4 — Tag and push

```bash
# Replace 123456789012 with your AWS account ID
ECR_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/employment-forecasting

docker tag employment-forecasting:latest $ECR_URI:latest

docker push $ECR_URI:latest
```

### Step 2.5 — Verify push

```bash
aws ecr list-images --repository-name employment-forecasting --region us-east-1
```

---

## Part 3: Deploy on AWS ECS (Fargate) — Serverless

### Step 3.1 — Create ECS cluster

```bash
aws ecs create-cluster --cluster-name employment-cluster --region us-east-1
```

### Step 3.2 — Create task definition

Create file `task-definition.json`:

```json
{
  "family": "employment-forecasting-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "employment-api",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/employment-forecasting:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/employment-forecasting",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register it:
```bash
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json \
  --region us-east-1
```

### Step 3.3 — Create security group

```bash
# Get your VPC ID first
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region us-east-1)

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name employment-sg \
  --description "Employment Forecasting API SG" \
  --vpc-id $VPC_ID \
  --region us-east-1 \
  --query "GroupId" --output text)

# Allow inbound on port 8000
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 \
  --region us-east-1

echo "Security Group: $SG_ID"
```

### Step 3.4 — Create ECS service

```bash
# Get subnet IDs
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" \
  --output text --region us-east-1 | tr '\t' ',')

aws ecs create-service \
  --cluster employment-cluster \
  --service-name employment-service \
  --task-definition employment-forecasting-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region us-east-1
```

### Step 3.5 — Get the public IP

```bash
# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster employment-cluster \
  --service-name employment-service \
  --query "taskArns[0]" --output text --region us-east-1)

# Get ENI attachment
ENI_ID=$(aws ecs describe-tasks \
  --cluster employment-cluster \
  --tasks $TASK_ARN \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
  --output text --region us-east-1)

# Get public IP
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI_ID \
  --query "NetworkInterfaces[0].Association.PublicIp" \
  --output text --region us-east-1)

echo "API available at: http://$PUBLIC_IP:8000"
echo "Swagger docs:     http://$PUBLIC_IP:8000/docs"
```

### Step 3.6 — Test the public endpoint

```bash
curl http://$PUBLIC_IP:8000/health
```

---

## Part 4: Deploy on AWS EC2 Ubuntu

### Step 4.1 — Launch EC2 instance

```bash
# Find Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text --region us-east-1)

# Create key pair (save the .pem file!)
aws ec2 create-key-pair \
  --key-name employment-key \
  --query "KeyMaterial" \
  --output text > employment-key.pem

chmod 400 employment-key.pem   # Mac/Linux only

# Launch instance (t3.medium recommended; t2.micro for free tier)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --count 1 \
  --instance-type t3.medium \
  --key-name employment-key \
  --security-group-ids $SG_ID \
  --associate-public-ip-address \
  --region us-east-1 \
  --query "Instances[0].InstanceId" \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1

# Get public DNS
EC2_DNS=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query "Reservations[0].Instances[0].PublicDnsName" \
  --output text --region us-east-1)

echo "Connect: ssh -i employment-key.pem ubuntu@$EC2_DNS"
```

### Step 4.2 — Connect via SSH

```bash
# Mac/Linux:
ssh -i employment-key.pem ubuntu@$EC2_DNS

# Windows (PowerShell):
ssh -i employment-key.pem ubuntu@$EC2_DNS

# Windows (PuTTY): Convert .pem to .ppk using PuTTYgen, then connect
```

### Step 4.3 — Install Docker on EC2

Run these commands on the EC2 instance (after SSH):

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
newgrp docker

# Verify
docker --version
```

### Step 4.4 — Install AWS CLI on EC2

```bash
sudo apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

### Step 4.5 — Configure AWS CLI on EC2

```bash
aws configure
# Enter your Access Key, Secret Key, region (us-east-1), output format (json)
```

### Step 4.6 — Pull image from ECR and run

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Pull image
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/employment-forecasting:latest

# Run container (restart always = survives reboots)
docker run -d \
  --name employment-api \
  --restart always \
  -p 8000:8000 \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/employment-forecasting:latest

# Check logs
docker logs employment-api -f
```

### Step 4.7 — Make it publicly accessible

Your EC2 public DNS is: `http://$EC2_DNS:8000`

```bash
# Test from EC2
curl http://localhost:8000/health

# Test from your laptop
curl http://$EC2_DNS:8000/health
```

Demo URL format: `http://ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com:8000/docs`

---

## Part 5: Cleanup (Avoid Ongoing Charges)

```bash
# 1. Stop ECS service
aws ecs update-service --cluster employment-cluster \
  --service employment-service --desired-count 0 --region us-east-1
aws ecs delete-service --cluster employment-cluster \
  --service employment-service --region us-east-1

# 2. Delete ECS cluster
aws ecs delete-cluster --cluster employment-cluster --region us-east-1

# 3. Delete ECR repo and images
aws ecr delete-repository \
  --repository-name employment-forecasting \
  --force --region us-east-1

# 4. Terminate EC2 instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1

# 5. Delete key pair
aws ec2 delete-key-pair --key-name employment-key --region us-east-1

# 6. Delete security group (after instances terminated)
aws ec2 delete-security-group --group-id $SG_ID --region us-east-1

# 7. Delete CloudWatch log group
aws logs delete-log-group \
  --log-group-name /ecs/employment-forecasting --region us-east-1
```

---

## DO / DO NOT

| DO | DO NOT |
|----|--------|
| Save your .pem key file securely | Share your AWS credentials publicly |
| Use t2.micro for free tier testing | Leave EC2 running when not in use |
| Tag all AWS resources for tracking | Forget to delete resources after testing |
| Test locally with Docker before pushing | Push images without testing locally |
| Enable EC2 security groups on port 8000 | Open port 22 to 0.0.0.0/0 (use your IP only) |
