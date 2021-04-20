pipeline {
    agent any
    options {
        timeout(time: 30, unit: 'MINUTES')
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                echo "${BUILD_ID}"
            }
        }
        stage('Test') {
            steps {
                echo 'testing'
            }
        }
    }
    post {
        always {
            echo 'post build'
        }
    }
}
