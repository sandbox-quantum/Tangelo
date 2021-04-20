pipeline {
    agent any
    options {
        timeout(time: 30, unit: 'MINUTES')
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh "docker build . -t qsdk_test_image"
            }
        }
        stage('Test') {
            steps {
                sh "docker image ls"
                sh "docker run qsdk_test_image ./cont_integration/run_test.sh"
            }
        }
    }
    post {
        always {
            echo "maybe remove qsdk_test_image"
        }
    }
}
