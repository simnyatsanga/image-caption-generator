import boto3
import argparse

def upload_model(s3_client, bucket_name, source_path):
    s3_object_key = source_path.split('/')[-1]
    print("Uploading: bucket_name => {}, source_path => {}, s3_object => {}".format(bucket_name, source_path, s3_object_key))
    s3_client.upload_file(source_path, bucket_name, s3_object_key)

def download_model(s3_client, bucket_name, s3_object_key):
    destination_path = 'keras/model_checkpoints' + s3_object_key
    print("Downloading: bucket_name => {}, s3_object_key => {}, destination_path => {}".format(bucket_name, s3_object_key, destination_path))
    s3_client.download_file(bucket_name, s3_object_key, destination_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op')
    parser.add_argument('--source_path')
    parser.add_argument('--model_name')
    parser.add_argument('--aws_access_key_id')
    parser.add_argument('--aws_secret_access_key')
    parser.add_argument('--bucket_name')

    args = parser.parse_args()

    if (not args.aws_access_key_id) or (not args.aws_secret_access_key):
        raise Exception('Please specify S3 credentials.')

    s3_client = boto3.client('s3', aws_access_key_id=args.aws_access_key_id, aws_secret_access_key=args.aws_secret_access_key)

    if args.op == 'upload' and args.source_path:
        upload_model(s3_client, args.bucket_name, args.source_path)
    elif args.op == 'upload' and not args.source_path:
        raise Exception('Please specify \'source_path\' for upload.')
    elif args.op == 'download' and args.model_name:
        download_model(s3_client, args.bucket_name, args.model_name)
    elif args.op == 'download' and not args.model_name:
        raise Exception('Please specify \'model_name\' for download.')
