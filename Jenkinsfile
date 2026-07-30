def builtImage

pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '5'))
    }

    parameters {
        string(name: 'DOCKER_IMAGE', defaultValue: 'automate-data-preprocessing', description: 'Docker image name without registry host')
        string(name: 'DOCKER_REGISTRY', defaultValue: '', description: 'Optional registry host, for example registry.example.com')
        string(name: 'DOCKER_REGISTRY_CREDENTIALS_ID', defaultValue: 'docker-registry-creds', description: 'Jenkins credentials id for registry login')
        string(name: 'DEPLOY_HOST', defaultValue: '', description: 'Optional remote host for SSH-based deployment')
        string(name: 'DEPLOY_USER', defaultValue: '', description: 'Optional SSH user for deployment')
        string(name: 'DEPLOY_SSH_CREDENTIALS_ID', defaultValue: 'ssh-deploy-key', description: 'Jenkins credentials id for SSH deploy access')
        string(name: 'DEPLOY_CONTAINER_NAME', defaultValue: 'automate-data-preprocessing', description: 'Container name to replace on deploy')
        string(name: 'DEPLOY_ENV_FILE', defaultValue: '/opt/automate-data-preprocessing/.env', description: 'Remote .env file mounted into the container')
    }

    environment {
        FULL_IMAGE_NAME = "${params.DOCKER_REGISTRY ? params.DOCKER_REGISTRY + '/' : ''}${params.DOCKER_IMAGE}:${env.BUILD_NUMBER}"
        LATEST_IMAGE_NAME = "${params.DOCKER_REGISTRY ? params.DOCKER_REGISTRY + '/' : ''}${params.DOCKER_IMAGE}:latest"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                script {
                    builtImage = docker.build(env.FULL_IMAGE_NAME)
                }
            }
        }

        stage('Tag Latest') {
            steps {
                script {
                    sh "docker tag ${env.FULL_IMAGE_NAME} ${env.LATEST_IMAGE_NAME}"
                }
            }
        }

        stage('Push Image') {
            when {
                expression { return params.DOCKER_REGISTRY?.trim() }
            }
            steps {
                script {
                    docker.withRegistry("https://${params.DOCKER_REGISTRY}", params.DOCKER_REGISTRY_CREDENTIALS_ID) {
                        builtImage.push()
                        sh "docker tag ${env.FULL_IMAGE_NAME} ${env.LATEST_IMAGE_NAME}"
                        sh "docker push ${env.LATEST_IMAGE_NAME}"
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                allOf {
                    expression { return params.DEPLOY_HOST?.trim() }
                    expression { return params.DEPLOY_USER?.trim() }
                }
            }
            steps {
                sshagent(credentials: [params.DEPLOY_SSH_CREDENTIALS_ID]) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ${params.DEPLOY_USER}@${params.DEPLOY_HOST} '
                            docker pull ${env.FULL_IMAGE_NAME} &&
                            docker stop ${params.DEPLOY_CONTAINER_NAME} || true &&
                            docker rm ${params.DEPLOY_CONTAINER_NAME} || true &&
                            docker run -d \
                                --name ${params.DEPLOY_CONTAINER_NAME} \
                                --restart unless-stopped \
                                -p 8501:8501 \
                                -v ${params.DEPLOY_ENV_FILE}:/app/.env:ro \
                                ${env.FULL_IMAGE_NAME}'
                    """
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}