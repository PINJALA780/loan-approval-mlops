import boto3
import sagemaker

REGION = "ap-south-1"

ROLE = "arn:aws:iam::540581805022:role/SageMakerExecutionRole"

BUCKET = "loan-approval-mlops-bucket"

SESSION = boto3.Session(region_name=REGION)

SAGEMAKER_SESSION = sagemaker.Session(
    boto_session=SESSION,
    default_bucket=BUCKET
)
