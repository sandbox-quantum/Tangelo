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
                sh "docker run qsdk_test_image ./cont_integration/run_test.sh"
            }
        }
    }
    post {
        always {
            echo "Cleaning up image and containers"
            sh "docker ps -a | awk '{ print \$1,\$2 }' | grep qsdk_test_image | awk '{print \$1 }' | xargs -I {} docker rm {}"
            sh "docker image rm qsdk_test_image"
        }
    }
}
